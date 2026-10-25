"""Extracción de características clásicas de una imagen (sin redes neuronales).

Cada imagen se convierte en un vector numérico que concatena:
- color: histograma de cada canal HSV y media/desviación de RGB
- hog:   histograma de gradientes orientados (forma y bordes)
- lbp:   patrones binarios locales (textura)
- color_espacial: media y desviación de HSV en una rejilla 3x3 y matiz-saturación conjunto del centro
  (el alimento suele estar al centro, así pesa menos el fondo)
- hog_fino: HOG con celdas de 8 px (más detalle de forma)
- lbp_multi: LBP a tres radios (textura fina y gruesa)
- forma: silueta del objeto separado del fondo por saturación (área, excentricidad, solidez...)
"""
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

TIPOS = ("color", "hog", "lbp")  # los de siempre; los modelos ya publicados dependen de este orden


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


def _color_espacial(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(rgb)
    lado = hsv.shape[0] // 3
    celdas = [hsv[i * lado:(i + 1) * lado, j * lado:(j + 1) * lado].reshape(-1, 3) for i in range(3) for j in range(3)]
    stats = np.concatenate([np.concatenate([c.mean(0), c.std(0)]) for c in celdas])
    centro = hsv[lado:2 * lado, lado:2 * lado].reshape(-1, 3)
    hs, _, _ = np.histogram2d(centro[:, 0], centro[:, 1], bins=(12, 4), range=((0, 1), (0, 1)))
    return np.concatenate([stats, hs.ravel() / max(len(centro), 1)])


def _hog_fino(gris: np.ndarray) -> np.ndarray:
    return hog(gris, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)


def _lbp_multi(gris: np.ndarray) -> np.ndarray:
    g = (gris * 255).astype(np.uint8)
    hists = []
    for puntos, radio in ((8, 1), (16, 2), (24, 3)):
        codigos = local_binary_pattern(g, P=puntos, R=radio, method="uniform")
        hists.append(np.histogram(codigos, bins=puntos + 2, range=(0, puntos + 2), density=True)[0])
    return np.concatenate(hists)


def _forma(rgb: np.ndarray) -> np.ndarray:
    """Separa el objeto del fondo por saturación (Otsu) y describe la región más grande."""
    sat = rgb2hsv(rgb)[..., 1]
    try:
        mascara = sat > threshold_otsu(sat)
    except ValueError:  # imagen de un solo color
        return np.zeros(7, dtype=np.float32)
    regiones = regionprops(label(mascara))
    if not regiones:
        return np.zeros(7, dtype=np.float32)
    r = max(regiones, key=lambda x: x.area)
    total = mascara.size
    alto, ancho = r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1]
    return np.array([r.area / total, mascara.mean(), r.eccentricity, r.solidity, r.extent,
                     alto / max(ancho, 1), r.perimeter ** 2 / (4 * np.pi * max(r.area, 1))], dtype=np.float32)


def extraer(imagen: Image.Image, tamano: int = 64, tipos=TIPOS, bins_color: int = 16) -> np.ndarray:
    rgb = np.asarray(imagen.convert("RGB").resize((tamano, tamano)), dtype=np.float32) / 255.0
    gris = rgb2gray(rgb)
    partes = {"color": lambda: _color(rgb, bins_color), "hog": lambda: _hog(gris), "lbp": lambda: _lbp(gris),
              "color_espacial": lambda: _color_espacial(rgb), "hog_fino": lambda: _hog_fino(gris),
              "lbp_multi": lambda: _lbp_multi(gris), "forma": lambda: _forma(rgb)}
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
