from collections import Counter

import numpy as np

from src.clasificador import Clasificador
from src.dataset import balancear_por_etiqueta, completar_por_clase, muestra_balanceada
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


def test_refuerzo_da_mas_imagenes_a_esos_alimentos():
    # 3 verduras con 100 imágenes; betabel (beetroot en el dataset) pesa el doble
    muestras = [(f"{c}{i}.jpg", 1, c) for c in ("beetroot", "lechuga", "kale") for i in range(100)]
    muestras += [(f"f{i}.jpg", 0, "manzana") for i in range(300)]
    train, test = muestra_balanceada(muestras, 120, 0.0, 0, refuerzo={"Betabel": 2})
    n = Counter(m[2] for m in train if m[1] == 1)
    assert n["beetroot"] == 60 and n["lechuga"] == 30 and n["kale"] == 30
    n = Counter(m[2] for m in balancear_por_etiqueta(muestras, 120, 0, {"betabel": 2}) if m[1] == 1)
    assert n["beetroot"] == 60 and n["lechuga"] == 30
    n = Counter(m[2] for m in completar_por_clase([], muestras, None, 0, 20, {"betabel": 2}) if m[1] == 1)
    assert n["beetroot"] == 40 and n["lechuga"] == 20


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


def test_metricas_traen_precision_y_recall_por_clase():
    from src.evaluate import metricas
    res = metricas([0, 0, 0, 1, 1], [0, 0, 1, 1, 0])
    assert res["precision_fruta"] == 2 / 3 and res["recall_fruta"] == 2 / 3
    assert res["precision_verdura"] == 0.5 and res["recall_verdura"] == 0.5
    assert res["recall_macro"] == res["balanced_accuracy"]


def test_intervalos_bootstrap_por_alimento():
    from src.evaluate import intervalos_bootstrap
    reales = [0] * 20 + [1] * 20
    predichos = [0] * 18 + [1] * 2 + [1] * 15 + [0] * 5
    grupos = [f"g{i // 5}" for i in range(40)]  # 8 alimentos de 5 fotos
    ic = intervalos_bootstrap(reales, predichos, grupos, 500, 0)
    assert ic["accuracy_ic_bajo"] <= 33 / 40 <= ic["accuracy_ic_alto"]
    assert 0 <= ic["f1_macro_ic_bajo"] <= ic["f1_macro_ic_alto"] <= 1
    # Sin errores el intervalo es exacto
    assert intervalos_bootstrap(reales, reales, grupos, 100, 0)["f1_macro_ic_bajo"] == 1.0
