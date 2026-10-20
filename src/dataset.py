"""Lectura de carpetas tipo ImageFolder (raiz/clase/imagen.jpg) con etiqueta binaria."""
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.mapping import ETIQUETAS, EXCLUIR, cargar_mapping, etiqueta_de, normalizar

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_muestras(raices: list[str], mapping: dict[str, str] | None = None) -> list[tuple[str, int, str]]:
    """Recorre cada raíz y devuelve (ruta, etiqueta, clase) con etiqueta 0 = fruta y 1 = verdura.

    Falla si alguna carpeta de clase no está en data/mapping.csv, para no entrenar con etiquetas faltantes.
    """
    mapping = mapping or cargar_mapping()
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


def limitar_por_clase(muestras: list, maximo: int | None, semilla: int) -> list:
    """Toma como máximo `maximo` imágenes de cada clase, para acotar memoria y tiempo."""
    if not maximo:
        return muestras
    rng = random.Random(semilla)
    por_clase = defaultdict(list)
    for m in muestras:
        por_clase[m[2]].append(m)
    return [m for grupo in por_clase.values() for m in rng.sample(grupo, min(maximo, len(grupo)))]


def conteo(muestras: list) -> dict[str, int]:
    c = Counter(m[1] for m in muestras)
    return {ETIQUETAS[y]: c.get(y, 0) for y in range(len(ETIQUETAS))}
