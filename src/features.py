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
- color_lab: media y desviación en Lab y histograma conjunto a-b (Lab separa mejor los tonos que HSV)
- bovw: "bolsa de palabras visuales": descriptores SIFT en una rejilla densa a 128 px, contados contra un
  vocabulario de patrones locales aprendido de las fotos de train (`aprender_vocabulario`)

Con `segmentar` ("saturacion" o "grabcut") primero se separa el alimento del fondo: el color se mide
solo sobre el alimento, el fondo se pinta de un gris liso antes de HOG y LBP, y la silueta sale de
esa misma máscara. Así el modelo no aprende el fondo (blanco de estudio, mesa, mercado).
"""
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import hog, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

TIPOS = ("color", "hog", "lbp")  # los de siempre; los modelos ya publicados dependen de este orden


def _color(rgb: np.ndarray, bins: int, m: np.ndarray | None = None) -> np.ndarray:
    hsv = rgb2hsv(rgb)
    px_hsv, px_rgb = (hsv[m], rgb[m]) if m is not None else (hsv.reshape(-1, 3), rgb.reshape(-1, 3))
    hists = [np.histogram(px_hsv[:, c], bins=bins, range=(0, 1), density=True)[0] / bins for c in range(3)]
    stats = np.concatenate([px_rgb.mean(0), px_rgb.std(0)])
    return np.concatenate(hists + [stats])


def _hog(gris: np.ndarray) -> np.ndarray:
    return hog(gris, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)


def _lbp(gris: np.ndarray) -> np.ndarray:
    puntos = 8
    codigos = local_binary_pattern((gris * 255).astype(np.uint8), P=puntos, R=1, method="uniform")
    return np.histogram(codigos, bins=puntos + 2, range=(0, puntos + 2), density=True)[0]


def _color_espacial(rgb: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
    hsv = rgb2hsv(rgb)
    m = np.ones(hsv.shape[:2], bool) if m is None else m
    lado = hsv.shape[0] // 3

    def pixeles(i, j):
        zona = (slice(i * lado, (i + 1) * lado), slice(j * lado, (j + 1) * lado))
        return hsv[zona][m[zona]]

    stats = []
    for i in range(3):
        for j in range(3):
            c = pixeles(i, j)
            stats.append(np.concatenate([c.mean(0), c.std(0)]) if len(c) else np.zeros(6))
    centro = pixeles(1, 1)
    hs, _, _ = np.histogram2d(centro[:, 0], centro[:, 1], bins=(12, 4), range=((0, 1), (0, 1)))
    return np.concatenate([np.concatenate(stats), hs.ravel() / max(len(centro), 1)])


def _hog_fino(gris: np.ndarray) -> np.ndarray:
    return hog(gris, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)


def _lbp_multi(gris: np.ndarray) -> np.ndarray:
    g = (gris * 255).astype(np.uint8)
    hists = []
    for puntos, radio in ((8, 1), (16, 2), (24, 3)):
        codigos = local_binary_pattern(g, P=puntos, R=radio, method="uniform")
        hists.append(np.histogram(codigos, bins=puntos + 2, range=(0, puntos + 2), density=True)[0])
    return np.concatenate(hists)


def _mascara_saturacion(rgb: np.ndarray) -> np.ndarray | None:
    """Píxeles más saturados que el umbral de Otsu (los fondos suelen ser grises o blancos)."""
    sat = rgb2hsv(rgb)[..., 1]
    try:
        return sat > threshold_otsu(sat)
    except ValueError:  # imagen de un solo color
        return None


def _mascara_grabcut(rgb: np.ndarray, iteraciones: int = 3) -> np.ndarray | None:
    """GrabCut de OpenCV: parte de la máscara por saturación (o del centro si no hay) y la refina
    con modelos de color del alimento y del fondo; el borde de la imagen se toma como fondo."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("segmentar='grabcut' necesita OpenCV: pip install opencv-python-headless") from e
    img = (rgb * 255).astype(np.uint8)
    lado = img.shape[0]
    marco = max(2, lado // 16)
    inicial = np.full(img.shape[:2], cv2.GC_PR_BGD, np.uint8)
    previa = _mascara_saturacion(rgb)
    if previa is not None and 0.05 < previa.mean() < 0.95:
        inicial[previa] = cv2.GC_PR_FGD
    else:
        inicial[lado // 4:3 * lado // 4, lado // 4:3 * lado // 4] = cv2.GC_PR_FGD
    inicial[:marco], inicial[-marco:], inicial[:, :marco], inicial[:, -marco:] = (cv2.GC_BGD,) * 4
    fondo, objeto = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), inicial, None, fondo, objeto, iteraciones,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:  # sin píxeles de una de las dos clases
        return previa
    return (inicial == cv2.GC_FGD) | (inicial == cv2.GC_PR_FGD)


def mascara(rgb: np.ndarray, metodo: str) -> np.ndarray | None:
    """Máscara del alimento (región conectada más grande), o None si no se pudo separar del fondo
    (casi todo o casi nada quedó como alimento): entonces se usa la imagen completa."""
    m = {"saturacion": _mascara_saturacion, "grabcut": _mascara_grabcut}[metodo](rgb)
    if m is None or not 0.03 < m.mean() < 0.97:
        return None
    etiquetas = label(m)
    mayor = max(regionprops(etiquetas), key=lambda r: r.area).label
    return etiquetas == mayor


def _forma(rgb: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
    """Describe la región más grande del objeto: la máscara dada o, si no hay, saturación con Otsu."""
    if m is None:
        m = _mascara_saturacion(rgb)
        if m is None:
            return np.zeros(7, dtype=np.float32)
    regiones = regionprops(label(m))
    if not regiones:
        return np.zeros(7, dtype=np.float32)
    r = max(regiones, key=lambda x: x.area)
    total = m.size
    alto, ancho = r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1]
    return np.array([r.area / total, m.mean(), r.eccentricity, r.solidity, r.extent,
                     alto / max(ancho, 1), r.perimeter ** 2 / (4 * np.pi * max(r.area, 1))], dtype=np.float32)


def _color_lab(rgb: np.ndarray, m: np.ndarray | None = None, bins: int = 8) -> np.ndarray:
    px = rgb2lab(rgb)[m] if m is not None else rgb2lab(rgb).reshape(-1, 3)
    ab, _, _ = np.histogram2d(px[:, 1], px[:, 2], bins=bins, range=((-80, 80), (-80, 80)))
    return np.concatenate([px.mean(0) / 100, px.std(0) / 100, ab.ravel() / len(px)])


LADO_SIFT, PASO_SIFT = 128, 8


def descriptores_sift(imagen: Image.Image) -> np.ndarray:
    """Descriptores SIFT (128 valores) en una rejilla densa de la imagen en gris a 128x128: 225 por imagen."""
    import cv2
    gris = np.asarray(imagen.convert("L").resize((LADO_SIFT, LADO_SIFT)))
    puntos = [cv2.KeyPoint(float(x), float(y), 2 * PASO_SIFT)
              for y in range(PASO_SIFT, LADO_SIFT, PASO_SIFT) for x in range(PASO_SIFT, LADO_SIFT, PASO_SIFT)]
    return cv2.SIFT_create().compute(gris, puntos)[1].astype(np.float32)


def _bovw(imagen: Image.Image, vocabulario: np.ndarray) -> np.ndarray:
    d = descriptores_sift(imagen)
    distancias = (d ** 2).sum(1)[:, None] - 2 * d @ vocabulario.T + (vocabulario ** 2).sum(1)[None]
    h = np.bincount(distancias.argmin(1), minlength=len(vocabulario)).astype(np.float32)
    return np.sqrt(h / h.sum())  # raíz de las frecuencias: que unas pocas palabras comunes no dominen


def _descriptores_ruta(ruta: str, n: int, semilla: int) -> np.ndarray:
    with Image.open(ruta) as imagen:
        d = descriptores_sift(imagen)
    return d[np.random.default_rng(semilla).choice(len(d), min(n, len(d)), replace=False)]


def aprender_vocabulario(rutas: list[str], palabras: int = 256, por_imagen: int = 40, max_imagenes: int = 3000,
                         semilla: int = 42) -> np.ndarray:
    """Vocabulario para "bovw": agrupa con k-means descriptores SIFT tomados de hasta `max_imagenes` fotos
    de train. Se guarda con el modelo, así que al predecir se usa el mismo."""
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(semilla)
    elegidas = [rutas[i] for i in rng.choice(len(rutas), min(max_imagenes, len(rutas)), replace=False)]
    d = np.vstack(Parallel(n_jobs=-1, batch_size=32)(delayed(_descriptores_ruta)(r, por_imagen, semilla)
                                                     for r in elegidas))
    km = MiniBatchKMeans(palabras, random_state=semilla, n_init=3, batch_size=4096).fit(d)
    return km.cluster_centers_.astype(np.float32)


def extraer(imagen: Image.Image, tamano: int = 64, tipos=TIPOS, bins_color: int = 16,
            segmentar: str | None = None, vocabulario: np.ndarray | None = None) -> np.ndarray:
    rgb = np.asarray(imagen.convert("RGB").resize((tamano, tamano)), dtype=np.float32) / 255.0
    gris = rgb2gray(rgb)
    m = mascara(rgb, segmentar) if segmentar else None
    if m is not None:
        gris = np.where(m, gris, gris[m].mean())  # fondo liso: sin bordes ni textura fuera del alimento
    partes = {"color": lambda: _color(rgb, bins_color, m), "hog": lambda: _hog(gris), "lbp": lambda: _lbp(gris),
              "color_espacial": lambda: _color_espacial(rgb, m), "hog_fino": lambda: _hog_fino(gris),
              "lbp_multi": lambda: _lbp_multi(gris), "forma": lambda: _forma(rgb, m),
              "color_lab": lambda: _color_lab(rgb, m), "bovw": lambda: _bovw(imagen, vocabulario)}
    if "bovw" in tipos and vocabulario is None:
        raise ValueError('"bovw" necesita un vocabulario (aprender_vocabulario con fotos de train)')
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
