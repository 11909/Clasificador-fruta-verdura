"""Predicción sobre una imagen.  Uso:  python -m src.predict outputs/modelo.joblib foto.jpg"""
import sys

import joblib
from PIL import Image

from src.features import extraer
from src.mapping import ETIQUETAS


def cargar(ruta_modelo: str) -> dict:
    return joblib.load(ruta_modelo)


def predecir(paquete: dict, imagen: Image.Image) -> dict[str, float]:
    x = extraer(imagen, **paquete["features"]).reshape(1, -1)
    modelo = paquete["modelo"]
    if hasattr(modelo, "predict_proba"):
        return {e: float(p) for e, p in zip(ETIQUETAS, modelo.predict_proba(x)[0])}
    return {e: float(i == modelo.predict(x)[0]) for i, e in enumerate(ETIQUETAS)}


if __name__ == "__main__":
    print(predecir(cargar(sys.argv[1]), Image.open(sys.argv[2])))
