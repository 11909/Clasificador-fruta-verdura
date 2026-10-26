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


def test_caracteristicas_nuevas():
    tipos = ["color", "color_espacial", "hog_fino", "lbp_multi", "forma"]
    a = extraer(imagen_aleatoria(0), tamano=64, tipos=tipos)
    b = extraer(imagen_aleatoria(1).resize((150, 80)), tamano=64, tipos=tipos)
    assert a.shape == b.shape and np.isfinite(a).all()
    # Una imagen lisa no tiene objeto que separar del fondo: la forma queda en ceros
    lisa = extraer(Image.new("RGB", (50, 50), (120, 60, 30)), tamano=64, tipos=["forma"])
    assert lisa.shape == (7,) and not lisa.any()


def test_validacion_deja_uno_fuera():
    from src.train import crear_cv, evaluar_cv
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40, 4)), rng.normal(3, 1, (40, 4))])
    y = np.array([0] * 40 + [1] * 40)
    grupos = np.repeat(np.arange(8), 10)  # 4 clases de fruta y 4 de verdura
    cv = crear_cv({"tipo": "uno_fuera_grupo"}, 0)
    assert cv.get_n_splits(X, y, grupos) == 8
    res, _ = evaluar_cv(crear_modelo("logistic_regression"), X, y, grupos, cv, deja_uno=True)
    assert res["balanced_accuracy"] > 0.9 and res["f1_macro_std"] == 0
    assert crear_cv({"tipo": "uno_fuera"}, 0).get_n_splits(X) == 80


def test_segmentar_separa_el_alimento_del_fondo():
    from PIL import ImageDraw
    from src.features import mascara
    im = Image.new("RGB", (120, 120), (240, 240, 235))
    ImageDraw.Draw(im).ellipse([30, 25, 95, 100], fill=(200, 40, 30))
    rgb = np.asarray(im.resize((64, 64)), dtype=np.float32) / 255
    tipos = ["color", "color_espacial", "hog_fino", "lbp_multi", "forma"]
    import importlib.util
    metodos = ["saturacion"] + (["grabcut"] if importlib.util.find_spec("cv2") else [])
    for metodo in metodos:
        assert abs(mascara(rgb, metodo).mean() - 0.27) < 0.03  # área de la elipse
        v = extraer(im, tipos=tipos, segmentar=metodo)
        assert v.shape == extraer(im, tipos=tipos).shape and np.isfinite(v).all()
    # Sin objeto que separar se usa la imagen completa
    assert mascara(np.full((64, 64, 3), 0.5, np.float32), metodos[-1]) is None


def test_validacion_anidada_busca_hiperparametros_dentro_de_cada_vuelta():
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from src.train import crear_cv, evaluar_cv
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 4)), rng.normal(2, 1, (60, 4))])
    y = np.array([0] * 60 + [1] * 60)
    grupos = np.repeat(np.arange(12), 10)
    externa = crear_cv({"tipo": "anidada", "folds": 3}, 0)
    interna = GridSearchCV(crear_modelo("knn"), {"kneighborsclassifier__n_neighbors": [3, 9]},
                           cv=StratifiedGroupKFold(n_splits=2), scoring="f1_macro")
    res, cvr = evaluar_cv(interna, X, y, grupos, externa, False, params={"groups": grupos})
    assert len(cvr["estimator"]) == 3 and all(hasattr(e, "best_params_") for e in cvr["estimator"])
    assert res["balanced_accuracy"] > 0.8 and "f1_macro_std" in res


def test_cache_de_pca_da_el_mismo_resultado(tmp_path):
    from sklearn.model_selection import GridSearchCV
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 8)), rng.normal(1, 1, (60, 8))])
    y = np.array([0] * 60 + [1] * 60)
    grid = {"kneighborsclassifier__n_neighbors": [3, 9]}
    sin = GridSearchCV(crear_modelo("knn", pca=4), grid, cv=3).fit(X, y)
    con = GridSearchCV(crear_modelo("knn", pca=4, cache=str(tmp_path)), grid, cv=3).fit(X, y)
    assert sin.best_params_ == con.best_params_ and (sin.predict(X) == con.predict(X)).all()
    assert any(tmp_path.iterdir())  # el escalado y el PCA quedaron guardados
    assert crear_modelo("knn", cache=str(tmp_path)).memory is None  # sin PCA no se guarda nada


def test_separar_vistos_aparta_fotos_de_cada_alimento():
    from src.train import separar_vistos
    grupos = np.repeat(["ajo", "pera", "kiwi"], [10, 5, 1])
    val = separar_vistos(grupos, 0.2, 0)
    assert val[grupos == "ajo"].sum() == 2 and val[grupos == "pera"].sum() == 1 and val[grupos == "kiwi"].sum() == 0
