"""Combina varios datasets de imágenes en un dataset propio con estructura única:

    salida/train/<clase>/<fuente>_<hash>.jpg
    salida/test/<clase>/<fuente>_<hash>.jpg
    salida/manifest.csv       (origen de cada imagen)
    salida/reporte.json       (conteos, duplicados y clases sin mapeo por fuente)

Cada fuente se describe en configs/dataset.yaml (dónde están sus splits, en qué nivel de carpeta
está la clase y cómo limpiar sus nombres). Las imágenes repetidas se eliminan por hash, y una imagen
que aparece en test nunca se copia a train.

Uso:
    python -m src.build_dataset --config configs/dataset.yaml
    python -m src.build_dataset --explorar /kaggle/input/algun-dataset   # para ver su estructura
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

from src.dataset import EXTENSIONES
from src.mapping import EXCLUIR, cargar_mapping, etiqueta_de, normalizar

SPLITS = ("test", "train")  # test primero: si una imagen está en ambos, se queda solo en test


def resolver_clase(nombre_carpeta: str, fuente: dict, mapping: dict) -> tuple[str, str | None]:
    """Limpia el nombre de carpeta de una fuente y lo busca en el mapeo.

    Devuelve (clase_limpia, etiqueta); etiqueta es None si no está en data/mapping.csv.
    """
    nombre = nombre_carpeta.lower().replace("_", " ").replace("-", " ").strip()
    for prefijo in fuente.get("quitar_prefijos", []):
        # cubre "fresh apple" y también "freshapples"
        if nombre.startswith(prefijo):
            nombre = nombre[len(prefijo):].strip()
            break
    nombre = fuente.get("renombrar", {}).get(nombre, nombre)
    nombre = normalizar(nombre)
    # Prueba el nombre tal cual y luego en singular ("apples" -> "apple", "tomatoes" -> "tomato")
    for candidato in (nombre, re.sub(r"es$", "", nombre), re.sub(r"s$", "", nombre)):
        etiqueta = etiqueta_de(candidato, mapping)
        if etiqueta is not None:
            return candidato, etiqueta
    return nombre, None


def listar_fuente(fuente: dict, split: str, mapping: dict, semilla: int) -> tuple[list[dict], Counter, Counter]:
    """Lista las imágenes de un split de una fuente con su clase y etiqueta."""
    raiz = Path(fuente["splits"][split])
    nivel = fuente.get("nivel_clase", 0)
    por_clase, sin_mapeo, excluidas = defaultdict(list), Counter(), Counter()
    for ruta in sorted(raiz.rglob("*")):
        if ruta.suffix.lower() not in EXTENSIONES or not ruta.is_file():
            continue
        partes = ruta.relative_to(raiz).parts
        if len(partes) <= nivel + 1:
            continue
        clase, etiqueta = resolver_clase(partes[nivel], fuente, mapping)
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
    mapping = cargar_mapping()
    vistas, manifest, reporte = set(), [], {}

    for split in SPLITS:
        for fuente in cfg["fuentes"]:
            if split not in fuente["splits"]:
                continue
            nombre = fuente["nombre"]
            muestras, sin_mapeo, excluidas = listar_fuente(fuente, split, mapping, semilla)
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
    faltantes = sorted({c for r in reporte.values() for c in r["sin_mapeo"]})
    if faltantes:
        print(f"\nCarpetas sin entrada en data/mapping.csv (no se copiaron): {faltantes}")
        print("Agrégalas al CSV (o a 'renombrar' en la config) y vuelve a correr.")
    return reporte


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
    args = parser.parse_args()
    if args.explorar:
        explorar(args.explorar)
    else:
        construir(yaml.safe_load(open(args.config, encoding="utf-8")))
