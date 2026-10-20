import numpy as np
from PIL import Image

from src.features import extraer
from src.models import crear_modelo


def imagen_aleatoria(semilla=0):
    rng = np.random.default_rng(semilla)
    return Image.fromarray((rng.random((100, 100, 3)) * 255).astype("uint8"))


def test_vector_de_tamano_fijo():
    a = extraer(imagen_aleatoria(0))
    b = extraer(imagen_aleatoria(1).resize((150, 80)))
    assert a.shape == b.shape and a.ndim == 1
    assert np.isfinite(a).all()


def test_tipos_de_caracteristicas_seleccionables():
    solo_color = extraer(imagen_aleatoria(), tipos=["color"], bins_color=16)
    assert solo_color.shape == (16 * 3 + 6,)


def test_todos_los_modelos_entrenan():
    rng = np.random.default_rng(0)
    X, y = rng.random((60, 10)), np.array([0, 1] * 30)
    for nombre in ["logistic_regression", "svm", "knn", "naive_bayes", "decision_tree",
                   "random_forest", "gradient_boosting"]:
        assert crear_modelo(nombre).fit(X, y).predict(X).shape == (60,)
    combinado = {"estimadores": {"logistic_regression": {}, "random_forest": {"n_estimators": 10}}}
    for nombre in ["voting", "stacking"]:
        assert crear_modelo(nombre, combinado, pca=5).fit(X, y).predict_proba(X).shape == (60, 2)
