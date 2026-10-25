"""Limpieza del dataset combinado al cargarlo, sin tocar el dataset de Kaggle.

Fruits-360 son cuadros de video de una pieza girando sobre fondo blanco: cientos de fotos casi iguales
por variedad, que inflan algunas clases y le enseñan al modelo el fondo y no el alimento. Aquí se
quitan los cuadros casi duplicados (por hash perceptual), se deja un tope de fotos de Fruits-360 por
clase y, si se pide, se descartan las fotos de cualquier fuente con fondo blanco liso.
"""
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image

PREFIJO_FRUITS360 = "fruits360_"  # build_dataset nombra cada imagen <fuente>_<hash>.jpg


def es_fruits360(ruta: str) -> bool:
    return Path(ruta).name.startswith(PREFIJO_FRUITS360)


def _abrir(ruta: str) -> Image.Image:
    with Image.open(ruta) as im:
        return im.convert("RGB")


def dhash(ruta: str, lado: int = 8) -> int:
    """Hash perceptual de 64 bits: compara cada píxel con su vecino en una versión gris de 9x8. Dos
    fotos casi iguales (otro ángulo de la misma pieza, otra compresión) difieren en pocos bits."""
    gris = np.asarray(_abrir(ruta).convert("L").resize((lado + 1, lado)), dtype=np.int16)
    bits = (gris[:, 1:] > gris[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def fondo_blanco(ruta: str, claro: int = 230, fraccion: float = 0.9) -> bool:
    """True si al menos `fraccion` del borde (5 % de cada lado) es casi blanco: foto de estudio."""
    a = np.asarray(_abrir(ruta).resize((64, 64)))
    borde = np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3), a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)])
    return (borde.min(axis=1) >= claro).mean() >= fraccion


def _distintos(grupo: list, umbral: int, tope: int | None, lote: int = 512) -> list:
    """Recorre `grupo` en orden y descarta cada imagen que difiere en `umbral` bits o menos de una ya
    elegida. Calcula los hashes por lotes y para al llegar a `tope`, así una clase con miles de
    cuadros no obliga a abrirlos todos."""
    elegidas, vistos = [], np.zeros(0, dtype=np.uint64)
    for inicio in range(0, len(grupo), lote):
        parte = grupo[inicio:inicio + lote]
        hashes = Parallel(n_jobs=-1, batch_size=64)(delayed(dhash)(m[0]) for m in parte)
        for m, h in zip(parte, hashes):
            if len(vistos):
                diferencia = np.bitwise_xor(vistos, np.uint64(h))
                if np.unpackbits(diferencia.view(np.uint8)).reshape(len(vistos), 64).sum(axis=1).min() <= umbral:
                    continue
            elegidas.append(m)
            vistos = np.append(vistos, np.uint64(h))
            if tope and len(elegidas) >= tope:
                return elegidas
    return elegidas


def limpiar(muestras: list, tope_fruits360: int | None = 100, umbral_duplicado: int | None = 6,
            quitar_fondo_blanco: bool = False, semilla: int = 42) -> list:
    """Filtra (ruta, etiqueta, clase):

    - `umbral_duplicado`: quita fotos de Fruits-360 casi iguales a otra de su clase (bits de diferencia
      en el hash; None = no quitar).
    - `tope_fruits360`: deja como máximo esas fotos de Fruits-360 por clase (None = sin tope).
    - `quitar_fondo_blanco`: descarta las fotos de cualquier fuente con fondo blanco liso. Todo
      Fruits-360 tiene fondo blanco, así que esto lo saca completo.
    """
    rng = random.Random(semilla)
    f360 = defaultdict(list)
    otras = []
    for m in muestras:
        (f360[m[2]] if es_fruits360(m[0]) else otras).append(m)
    antes = sum(len(g) for g in f360.values())

    limpias = []
    for clase in sorted(f360):
        grupo = sorted(f360[clase])
        rng.shuffle(grupo)  # el tope se queda con una muestra al azar, no con los primeros cuadros
        if umbral_duplicado is not None:
            limpias += _distintos(grupo, umbral_duplicado, tope_fruits360)
        else:
            limpias += grupo[:tope_fruits360] if tope_fruits360 else grupo
    if antes:
        print(f"limpieza Fruits-360: {antes} -> {len(limpias)} imágenes en {len(f360)} clases")

    salida = otras + limpias
    if quitar_fondo_blanco:
        blancas = Parallel(n_jobs=-1, batch_size=64)(delayed(fondo_blanco)(m[0]) for m in salida)
        print(f"limpieza fondo blanco: se quitan {sum(blancas)} de {len(salida)}")
        salida = [m for m, b in zip(salida, blancas) if not b]
    return salida
