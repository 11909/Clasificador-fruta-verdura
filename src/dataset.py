"""Lectura de carpetas tipo ImageFolder (raiz/clase/imagen.jpg) con etiqueta binaria."""
import random
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.mapping import ETIQUETAS, EXCLUIR, cargar_mapping, etiqueta_de

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def listar_muestras(raices: list[str], mapping: dict[str, str] | None = None) -> list[tuple[str, int]]:
    """Recorre cada raíz y devuelve (ruta, etiqueta) con 0 = fruta y 1 = verdura.

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
            y = ETIQUETAS.index(etiqueta)
            muestras += [(str(p), y) for p in sorted(carpeta.iterdir()) if p.suffix.lower() in EXTENSIONES]
    if sin_mapeo:
        raise ValueError(f"Clases sin entrada en data/mapping.csv: {sorted(sin_mapeo)}")
    return muestras


def dividir(muestras: list, fraccion_val: float, semilla: int) -> tuple[list, list]:
    """División aleatoria estratificada por etiqueta."""
    rng = random.Random(semilla)
    train, val = [], []
    for y in range(len(ETIQUETAS)):
        grupo = [m for m in muestras if m[1] == y]
        rng.shuffle(grupo)
        corte = int(len(grupo) * fraccion_val)
        val += grupo[:corte]
        train += grupo[corte:]
    return train, val


def conteo(muestras: list) -> dict[str, int]:
    c = Counter(y for _, y in muestras)
    return {ETIQUETAS[y]: c.get(y, 0) for y in range(len(ETIQUETAS))}


def transformaciones(tamano: int, entrenamiento: bool):
    if entrenamiento:
        return transforms.Compose([
            transforms.RandomResizedCrop(tamano, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((tamano, tamano)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImagenesBinarias(Dataset):
    def __init__(self, muestras: list[tuple[str, int]], transform):
        self.muestras = muestras
        self.transform = transform

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        ruta, y = self.muestras[i]
        return self.transform(Image.open(ruta).convert("RGB")), y
