"""Predicción sobre una imagen.  Uso:  python -m src.predict outputs/best.pt foto.jpg"""
import sys

import torch
from PIL import Image

from src.dataset import transformaciones
from src.mapping import ETIQUETAS
from src.model import crear_modelo


def cargar(checkpoint: str, device: str = "cpu"):
    ck = torch.load(checkpoint, map_location=device)
    modelo = crear_modelo(ck["model"], preentrenado=False)
    modelo.load_state_dict(ck["state_dict"])
    return modelo.to(device).eval(), transformaciones(ck["image_size"], entrenamiento=False)


@torch.no_grad()
def predecir(modelo, transform, imagen: Image.Image) -> dict[str, float]:
    probs = modelo(transform(imagen.convert("RGB")).unsqueeze(0)).softmax(1)[0]
    return {e: float(p) for e, p in zip(ETIQUETAS, probs)}


if __name__ == "__main__":
    modelo, transform = cargar(sys.argv[1])
    print(predecir(modelo, transform, Image.open(sys.argv[2])))
