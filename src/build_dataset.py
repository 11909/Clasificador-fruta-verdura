"""Combina varios datasets de imágenes en un dataset propio con estructura única:

    salida/train/<clase>/<fuente>_<hash>.jpg
    salida/test/<clase>/<fuente>_<hash>.jpg
    salida/manifest.csv       (origen de cada imagen)
    salida/conteos.csv        (imágenes por etiqueta, clase y fuente, para revisar el desbalance)
    salida/reporte.json       (conteos, duplicados y clases sin mapeo por fuente)

Cada fuente se describe en configs/dataset.yaml (dónde están sus splits, en qué nivel de carpeta
está la clase y cómo limpiar sus nombres). Las imágenes repetidas se eliminan por hash, y una imagen
que aparece en test nunca se copia a train.

Las clases válidas son las de data/mapping.csv: ahí se define cada fruta y verdura, y una carpeta
por alimento basta para que la fuente del equipo entre al dataset.

Uso:
    python -m src.build_dataset --config configs/dataset.yaml
    python -m src.build_dataset --explorar /kaggle/input/algun-dataset   # para ver su estructura
    python -m src.build_dataset --plantilla fotos_equipo                 # una carpeta vacía por alimento
"""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed
from PIL import Image

from src.dataset import EXTENSIONES, va_a_test
from src.mapping import EXCLUIR, MAPPING_PATH, cargar_alimentos, cargar_mapping, clave_de, leer_filas, normalizar

SPLITS = ("test", "train")  # test primero: si una imagen está en ambos, se queda solo en test


def resolver_clase(nombre_carpeta: str, fuente: dict, mapping: dict,
                   alimentos: dict | None = None) -> tuple[str, str | None]:
    """Limpia el nombre de carpeta de una fuente y lo busca en el mapeo.

    Devuelve (alimento en español, etiqueta), p. ej. "Apple Red 2" -> ("manzana", "fruta"). Si no está
    en data/mapping.csv devuelve (nombre limpio, None).
    """
    alimentos = alimentos or cargar_alimentos()
    nombre = nombre_carpeta.lower().replace("_", " ").replace("-", " ").strip()
    for prefijo in fuente.get("quitar_prefijos", []):
        # cubre "fresh apple" y también "freshapples"
        if nombre.startswith(prefijo):
            nombre = nombre[len(prefijo):].strip()
            break
    nombre = normalizar(nombre)
    nombre = normalizar(fuente.get("renombrar", {}).get(nombre, nombre))
    # Prueba el nombre tal cual y luego en singular ("apples" -> "apple", "tomatoes" -> "tomato")
    for candidato in (nombre, re.sub(r"es$", "", nombre), re.sub(r"s$", "", nombre)):
        clave = clave_de(candidato, mapping)
        if clave is not None:
            return alimentos[clave], mapping[clave]
    return nombre, None


def ubicar(raiz: Path, relativa: str) -> Path:
    """`raiz/relativa`; si no existe, la carpeta bajo `raiz` cuya ruta termina en `relativa` (algunos
    datasets de Kaggle traen un nivel extra, como dataset/dataset/Train)."""
    directa = raiz / relativa
    if directa.is_dir() or relativa in ("", "."):
        return directa
    candidatas = [p for p in raiz.rglob(Path(relativa).name)
                  if p.is_dir() and p.as_posix().endswith("/" + Path(relativa).as_posix())]
    return min(candidatas, key=lambda p: len(p.parts)) if candidatas else directa


def preparar_fuentes(cfg: dict, descargar=None) -> dict:
    """Para cada fuente con `kaggle: dueño/dataset`, la descarga (o monta, dentro de Kaggle) con
    kagglehub y vuelve absolutas sus rutas, que en la config son relativas a la raíz del dataset.
    Una fuente que no se puede descargar se quita con un aviso. Devuelve una copia de la config."""
    if descargar is None:
        import kagglehub
        descargar = kagglehub.dataset_download
    fuentes = []
    for fuente in cfg["fuentes"]:
        if "kaggle" not in fuente:
            fuentes.append(fuente)
            continue
        try:
            raiz = Path(descargar(fuente["kaggle"]))
        except Exception as e:  # dataset privado sin acceso, nombre mal escrito, sin Internet...
            print(f"Aviso: no se pudo descargar {fuente['kaggle']} ({fuente['nombre']}); se omite. {e}")
            continue
        fuente = {**fuente, "raiz": str(raiz)}
        absoluta = lambda r: [str(ubicar(raiz, x)) for x in r] if isinstance(r, list) else str(ubicar(raiz, r))
        if "ruta" in fuente:
            fuente["ruta"] = absoluta(fuente["ruta"])
        else:
            fuente["splits"] = {k: absoluta(v) for k, v in fuente["splits"].items()}
        print(f"{fuente['nombre']}: {raiz}")
        fuentes.append(fuente)
    return {**cfg, "fuentes": fuentes}


def splits_de(fuente: dict) -> set[str]:
    """Splits que aporta una fuente. Una fuente con `ruta` (sin splits propios) aporta a los dos."""
    return set(SPLITS) if "ruta" in fuente else set(fuente["splits"])


def rutas_de(fuente: dict, split: str) -> list[Path]:
    """Carpetas de un split; en la config puede ser una ruta o una lista (p. ej. train + validation)."""
    rutas = fuente["ruta"] if "ruta" in fuente else fuente["splits"][split]
    return [Path(r) for r in ([rutas] if isinstance(rutas, str) else rutas)]


def toca_split(ruta: Path, fuente: dict, split: str, semilla: int) -> bool:
    """Para fuentes sin splits propios: decide por hash del nombre si la imagen va a test, de modo que
    la asignación no cambia al agregar fotos nuevas."""
    if "ruta" not in fuente:
        return True
    return va_a_test(ruta, fuente.get("fraccion_test", 0.3), semilla) == (split == "test")


def listar_fuente(fuente: dict, split: str, mapping: dict, semilla: int,
                  alimentos: dict | None = None) -> tuple[list[dict], Counter, Counter]:
    """Lista las imágenes de un split de una fuente con su clase y etiqueta."""
    nivel = fuente.get("nivel_clase", 0)
    alimentos = alimentos or cargar_alimentos()
    ignorar = {c.lower() for c in fuente.get("ignorar", [])}
    por_clase, sin_mapeo, excluidas = defaultdict(list), Counter(), Counter()
    for raiz in rutas_de(fuente, split):
        if not raiz.is_dir():
            print(f"Aviso: no existe {raiz} ({fuente['nombre']}/{split}); se omite. ¿Está adjunta la fuente?")
            continue
        for ruta in sorted(raiz.rglob("*")):
            if ruta.suffix.lower() not in EXTENSIONES or not ruta.is_file():
                continue
            partes = ruta.relative_to(raiz).parts
            if len(partes) <= nivel + 1 or ignorar & {p.lower() for p in partes[:-1]}:
                continue
            if not toca_split(ruta, fuente, split, semilla):
                continue
            clase, etiqueta = resolver_clase(partes[nivel], fuente, mapping, alimentos)
            if etiqueta is None:
                sin_mapeo[partes[nivel]] += 1
            elif etiqueta == EXCLUIR:
                excluidas[clase] += 1
            else:
                por_clase[clase].append({"ruta_original": str(ruta), "clase": clase, "etiqueta": etiqueta})
    rng = random.Random(semilla)
    maximo = fuente.get("max_por_clase")
    elegidas = []
    for clase in sorted(por_clase):
        grupo = por_clase[clase]
        elegidas += rng.sample(grupo, maximo) if maximo and len(grupo) > maximo else grupo
    return elegidas, sin_mapeo, excluidas


def huella(ruta: str) -> str:
    """Hash del contenido de la imagen reducida a 64x64, para detectar la misma foto aunque
    venga con otro nombre o tamaño."""
    with Image.open(ruta) as img:
        pixeles = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.uint8)
    return hashlib.md5(pixeles.tobytes()).hexdigest()


def copiar(origen: str, destino: Path, tamano_max: int):
    with Image.open(origen) as img:
        img = img.convert("RGB")
        img.thumbnail((tamano_max, tamano_max))  # reduce sin deformar; nunca agranda
        destino.parent.mkdir(parents=True, exist_ok=True)
        img.save(destino, "JPEG", quality=90)


def construir(cfg: dict):
    salida = Path(cfg["salida"])
    semilla, tamano_max = cfg.get("seed", 42), cfg.get("tamano_max", 256)
    mapping, alimentos = cargar_mapping(), cargar_alimentos()
    vistas, manifest, reporte = set(), [], {}

    for split in SPLITS:
        for fuente in cfg["fuentes"]:
            if split not in splits_de(fuente):
                continue
            nombre = fuente["nombre"]
            muestras, sin_mapeo, excluidas = listar_fuente(fuente, split, mapping, semilla, alimentos)
            hashes = Parallel(n_jobs=-1, batch_size=64)(delayed(huella)(m["ruta_original"]) for m in muestras)
            nuevas, duplicadas = [], 0
            for m, h in zip(muestras, hashes):
                if h in vistas:
                    duplicadas += 1
                    continue
                vistas.add(h)
                destino = salida / split / m["clase"] / f"{nombre}_{h[:12]}.jpg"
                nuevas.append({**m, "split": split, "fuente": nombre, "hash": h, "ruta": str(destino.relative_to(salida))})
            Parallel(n_jobs=-1, batch_size=64)(
                delayed(copiar)(m["ruta_original"], salida / m["ruta"], tamano_max) for m in nuevas)
            manifest += nuevas
            reporte[f"{nombre}/{split}"] = {
                "copiadas": len(nuevas), "duplicadas": duplicadas,
                "por_etiqueta": dict(Counter(m["etiqueta"] for m in nuevas)),
                "clases": len({m["clase"] for m in nuevas}),
                "excluidas": dict(excluidas), "sin_mapeo": dict(sin_mapeo),
            }
            print(f"{nombre}/{split}: {len(nuevas)} copiadas, {duplicadas} duplicadas, "
                  f"{sum(sin_mapeo.values())} sin mapeo en {len(sin_mapeo)} carpetas")

    campos = ["ruta", "split", "clase", "etiqueta", "fuente", "ruta_original", "hash"]
    with open(salida / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(manifest)
    json.dump(reporte, open(salida / "reporte.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    escribir_conteos(manifest, salida / "conteos.csv")
    faltantes = sorted({c for r in reporte.values() for c in r["sin_mapeo"]})
    if faltantes:
        print(f"\nCarpetas sin entrada en data/mapping.csv (no se copiaron): {faltantes}")
        print("Agrégalas al CSV (o a 'renombrar' en la config) y vuelve a correr.")
    return reporte


def escribir_conteos(manifest: list[dict], destino: Path):
    """Tabla etiqueta, clase, fuente, train, test, ordenada por etiqueta y de la clase con más
    imágenes a la que tiene menos, e imprime el total por etiqueta."""
    conteo = Counter((m["etiqueta"], m["clase"], m["fuente"], m["split"]) for m in manifest)
    filas = defaultdict(lambda: {"train": 0, "test": 0})
    for (etiqueta, clase, fuente, split), n in conteo.items():
        filas[(etiqueta, clase, fuente)][split] += n
    por_clase = Counter()
    for (etiqueta, clase, _), n in filas.items():
        por_clase[(etiqueta, clase)] += n["train"] + n["test"]
    orden = sorted(filas, key=lambda k: (k[0], -por_clase[k[:2]], k[1], k[2]))
    with open(destino, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["etiqueta", "clase", "fuente", "train", "test"])
        writer.writerows([*k, filas[k]["train"], filas[k]["test"]] for k in orden)
    totales = Counter()
    for (etiqueta, _), n in por_clase.items():
        totales[etiqueta] += n
    clases = Counter(etiqueta for etiqueta, _ in por_clase)
    print("\nImágenes por etiqueta: " + ", ".join(
        f"{e} {totales[e]} en {clases[e]} clases" for e in sorted(totales)))


def crear_plantilla(destino: str, mapping_path: Path = MAPPING_PATH) -> list[str]:
    """Crea una carpeta vacía por cada alimento de data/mapping.csv (con su nombre en español, sin
    acentos), para que el equipo suba sus fotos en la que corresponde. Omite los excluidos. Un
    alimento nuevo se agrega primero al CSV."""
    nombres = sorted(normalizar(fila["alimento"]) for fila in leer_filas(mapping_path)
                     if fila["etiqueta"].strip() != EXCLUIR)
    for nombre in nombres:
        (Path(destino) / nombre.replace(" ", "_")).mkdir(parents=True, exist_ok=True)
    print(f"{len(nombres)} carpetas en {destino}")
    return nombres


def explorar(raiz: str, profundidad: int = 3, ejemplos: int = 8):
    """Imprime la estructura de carpetas para decidir splits y nivel_clase de una fuente nueva."""
    def recorrer(carpeta: Path, nivel: int):
        subcarpetas = sorted(p for p in carpeta.iterdir() if p.is_dir())
        imagenes = sum(1 for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
        sangria = "  " * nivel
        print(f"{sangria}{carpeta.name}/  ({len(subcarpetas)} carpetas, {imagenes} imágenes)")
        if nivel + 1 >= profundidad:
            return
        for sub in subcarpetas[:ejemplos]:
            recorrer(sub, nivel + 1)
        if len(subcarpetas) > ejemplos:
            print(f"{sangria}  ... y {len(subcarpetas) - ejemplos} carpetas más")
    recorrer(Path(raiz), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--explorar", help="carpeta a inspeccionar en lugar de construir")
    parser.add_argument("--plantilla", help="carpeta donde crear una subcarpeta vacía por alimento")
    args = parser.parse_args()
    if args.explorar:
        explorar(args.explorar)
    elif args.plantilla:
        crear_plantilla(args.plantilla)
    else:
        construir(yaml.safe_load(open(args.config, encoding="utf-8")))
