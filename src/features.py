"""Extracción de características clásicas de una imagen (sin redes neuronales).

Cada imagen se convierte en un vector numérico que concatena:
- color: histograma de cada canal HSV y media/desviación de RGB
- hog:   histograma de gradientes orientados (forma y bordes)
- lbp:   patrones binarios locales (textura)
"""
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern

TIPOS = ("color", "hog", "lbp")


def _color(rgb: np.ndarray, bins: int) -> np.ndarray:
    hsv = rgb2hsv(rgb)
    hists = [np.histogram(hsv[..., c], bins=bins, range=(0, 1), density=True)[0] / bins for c in range(3)]
    stats = np.concatenate([rgb.reshape(-1, 3).mean(0), rgb.reshape(-1, 3).std(0)])
    return np.concatenate(hists + [stats])


def _hog(gris: np.ndarray) -> np.ndarray:
    return hog(gris, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)


def _lbp(gris: np.ndarray) -> np.ndarray:
    puntos = 8
    codigos = local_binary_pattern((gris * 255).astype(np.uint8), P=puntos, R=1, method="uniform")
    return np.histogram(codigos, bins=puntos + 2, range=(0, puntos + 2), density=True)[0]


def extraer(imagen: Image.Image, tamano: int = 64, tipos=TIPOS, bins_color: int = 16) -> np.ndarray:
    rgb = np.asarray(imagen.convert("RGB").resize((tamano, tamano)), dtype=np.float32) / 255.0
    gris = rgb2gray(rgb)
    partes = {"color": lambda: _color(rgb, bins_color), "hog": lambda: _hog(gris), "lbp": lambda: _lbp(gris)}
    return np.concatenate([partes[t]() for t in tipos]).astype(np.float32)


def _extraer_ruta(ruta: str, **kwargs) -> np.ndarray:
    with Image.open(ruta) as imagen:
        return extraer(imagen, **kwargs)


def extraer_lote(rutas: list[str], n_jobs: int = -1, **kwargs) -> np.ndarray:
    """Extrae características de muchas imágenes en paralelo (usa todos los núcleos de CPU)."""
    filas = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_extraer_ruta)(r, **kwargs) for r in rutas)
    return np.vstack(filas)


def extraer_con_cache(rutas: list[str], cache: Path | None, **kwargs) -> np.ndarray:
    """Igual que extraer_lote, pero guarda el resultado en .npy para no repetir la extracción."""
    if cache and cache.exists():
        X = np.load(cache)
        if len(X) == len(rutas):
            return X
    X = extraer_lote(rutas, **kwargs)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
    return X
