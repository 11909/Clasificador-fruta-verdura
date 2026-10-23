import numpy as np
import pytest
from PIL import Image

from src.features import extraer
from src.modelo import guardar_modelo
from src.models import crear_modelo

FEATURES = {"tamano": 32, "tipos": ["color"], "bins_color": 8}


def imagen(color):
    return Image.new("RGB", (40, 40), color)


@pytest.fixture
def ruta_modelo(tmp_path):
    # rojo = fruta (0), verde = verdura (1)
    imgs = [imagen((200 + i, 10, 10)) for i in range(10)] + [imagen((10, 200 + i, 10)) for i in range(10)]
    X = np.vstack([extraer(i, **FEATURES) for i in imgs])
    y = np.array([0] * 10 + [1] * 10)
    modelo = crear_modelo("logistic_regression").fit(X, y)
    return guardar_modelo(modelo, FEATURES, {"name": "logistic_regression"}, tmp_path / "modelo.joblib")
