"""Lectura de carpetas tipo ImageFolder (raiz/clase/imagen.jpg) con etiqueta binaria."""
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.mapping import ETIQUETAS, EXCLUIR, cargar_mapping, etiqueta_de, normalizar

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_muestras(raices: list[str], mapping: dict[str, str] | None = None,
                    excluir_grupos=()) -> list[tuple[str, int, str]]:
    """Recorre cada raíz y devuelve (ruta, etiqueta, clase) con etiqueta 0 = fruta y 1 = verdura.

    Falla si alguna carpeta de clase no está en data/mapping.csv, para no entrenar con etiquetas faltantes.
    """
    mapping = mapping or cargar_mapping(excluir_grupos=excluir_grupos)
    muestras, sin_mapeo = [], set()
    for raiz in raices:
        for carpeta in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
            etiqueta = etiqueta_de(carpeta.name, mapping)
            if etiqueta is None:
                sin_mapeo.add(carpeta.name)
                continue
            if etiqueta == EXCLUIR:
                continue
            y, clase = ETIQUETAS.index(etiqueta), normalizar(carpeta.name)
            muestras += [(str(p), y, clase) for p in sorted(carpeta.iterdir()) if p.suffix.lower() in EXTENSIONES]
    if sin_mapeo:
        raise ValueError(f"Clases sin entrada en data/mapping.csv: {sorted(sin_mapeo)}")
    return muestras


def va_a_test(ruta, fraccion_test: float, semilla: int) -> bool:
    """Decide por hash de la carpeta y el nombre del archivo si una foto va a test, de modo que la
    asignación no cambia al agregar fotos nuevas ni entre versiones del dataset."""
    ruta = Path(ruta)
    clave = f"{semilla}/{ruta.parent.name}/{ruta.name}".encode()
    return int(hashlib.md5(clave).hexdigest(), 16) % 1000 < fraccion_test * 1000


def dividir_equipo(raiz: str, fraccion_test: float, semilla: int, excluir_grupos=()) -> tuple[list, list]:
    """Fotos del equipo (raiz/alimento/foto.jpg) repartidas entre train y test con va_a_test."""
    raiz = Path(raiz)
    if not raiz.is_dir():
        return [], []
    # Si el zip se subió con una carpeta envolvente (fotos/zanahoria/...), se entra en ella
    mapping = cargar_mapping(excluir_grupos=excluir_grupos)
    while True:
        subcarpetas = [p for p in raiz.iterdir() if p.is_dir()]
        if len(subcarpetas) != 1 or etiqueta_de(subcarpetas[0].name, mapping) is not None:
            break
        raiz = subcarpetas[0]
    muestras = listar_muestras([str(raiz)], mapping)
    test = [m for m in muestras if va_a_test(m[0], fraccion_test, semilla)]
    en_test = {m[0] for m in test}
    return [m for m in muestras if m[0] not in en_test], test


def limitar_por_clase(muestras: list, maximo: int | None, semilla: int) -> list:
    """Toma como máximo `maximo` imágenes de cada clase, para acotar memoria y tiempo."""
    if not maximo:
        return muestras
    rng = random.Random(semilla)
    por_clase = defaultdict(list)
    for m in muestras:
        por_clase[m[2]].append(m)
    return [m for grupo in por_clase.values() for m in rng.sample(grupo, min(maximo, len(grupo)))]


def balancear_por_etiqueta(muestras: list, maximo: int | None, semilla: int) -> list:
    """Deja el mismo número de imágenes de fruta y de verdura: el de la etiqueta con menos
    imágenes, o `maximo` si es menor. Dentro de cada etiqueta reparte por igual entre clases
    (una imagen de cada clase por turno) para no perder variedades."""
    rng = random.Random(semilla)
    por_etiqueta = defaultdict(lambda: defaultdict(list))
    for m in muestras:
        por_etiqueta[m[1]][m[2]].append(m)
    if len(por_etiqueta) < 2:
        return muestras
    objetivo = min(sum(map(len, clases.values())) for clases in por_etiqueta.values())
    if maximo:
        objetivo = min(objetivo, maximo)
    elegidas = []
    for clases in por_etiqueta.values():
        colas = [rng.sample(grupo, len(grupo)) for _, grupo in sorted(clases.items())]
        rng.shuffle(colas)
        tomadas = []
        while len(tomadas) < objetivo:
            for cola in colas:
                if cola and len(tomadas) < objetivo:
                    tomadas.append(cola.pop())
        elegidas += tomadas
    return elegidas


def conteo(muestras: list) -> dict[str, int]:
    c = Counter(m[1] for m in muestras)
    return {ETIQUETAS[y]: c.get(y, 0) for y in range(len(ETIQUETAS))}
