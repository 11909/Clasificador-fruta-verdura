from collections import Counter

import numpy as np

from src.clasificador import Clasificador
from src.dataset import balancear_por_etiqueta
from src.evaluate import elegir_umbral, predecir_con_umbral
from src.modelo import cargar_modelo


def test_balancear_por_etiqueta_iguala_y_reparte():
    # 3 clases de fruta con 50 imágenes y 1 de verdura con 20
    muestras = [(f"{c}{i}", 0, c) for c in "abc" for i in range(50)] + [(f"v{i}", 1, "v") for i in range(20)]
    elegidas = balancear_por_etiqueta(muestras, None, 0)
    assert Counter(m[1] for m in elegidas) == {0: 20, 1: 20}
    por_clase = Counter(m[2] for m in elegidas if m[1] == 0)
    assert set(por_clase) == {"a", "b", "c"} and max(por_clase.values()) - min(por_clase.values()) <= 1
    assert Counter(m[1] for m in balancear_por_etiqueta(muestras, 10, 0)) == {0: 10, 1: 10}
    assert len(set(elegidas)) == len(elegidas)


def test_elegir_umbral_corrige_sesgo_a_fruta():
    # Las verduras nunca pasan de 0.5, pero sí quedan por encima de las frutas
    y = np.array([0] * 90 + [1] * 10)
    p_verdura = np.concatenate([np.full(90, 0.1), np.full(10, 0.3)])
    probs = np.column_stack([1 - p_verdura, p_verdura])
    assert predecir_con_umbral(probs, 0.5).sum() == 0
    umbral = elegir_umbral(y, probs)
    assert 0.1 < umbral <= 0.3
    assert (predecir_con_umbral(probs, umbral) == y).all()


def test_clasificador_usa_umbral(ruta_modelo, tmp_path):
    from PIL import Image
    foto = tmp_path / "rojo.png"
    Image.new("RGB", (40, 40), (205, 10, 10)).save(foto)
    paquete = cargar_modelo(ruta_modelo)
    assert Clasificador(paquete).clasificar(str(foto))["etiqueta"] == "fruta"
    assert Clasificador({**paquete, "umbral": 0.0}).clasificar(str(foto))["etiqueta"] == "verdura"
