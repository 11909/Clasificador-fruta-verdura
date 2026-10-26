"""Catálogo de modelos clásicos de scikit-learn (sin redes neuronales).

Todos se envuelven en un Pipeline: escalado -> (PCA opcional) -> clasificador.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _base(nombre: str, params: dict, semilla: int):
    # class_weight="balanced" compensa que haya más imágenes de fruta que de verdura
    constructores = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", **params),
        # Calibración para obtener probabilidades (la demo y "voting" las usan)
        "svm": lambda: CalibratedClassifierCV(
            SVC(class_weight="balanced", random_state=semilla, **params), cv=3),
        "knn": lambda: KNeighborsClassifier(**params),
        "naive_bayes": lambda: GaussianNB(**params),
        "decision_tree": lambda: DecisionTreeClassifier(class_weight="balanced", random_state=semilla, **params),
        "random_forest": lambda: RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=semilla, **params),
        # Sin class_weight: train ya va balanceado (pesos de 1) y con pesos HistGradientBoosting tarda
        # unas 14 veces más. Si se entrena sin balancear, se puede pasar class_weight en params.
        "gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=semilla, **params),
    }
    if nombre not in constructores:
        raise ValueError(f"Modelo desconocido: {nombre}. Opciones: {sorted(constructores) + ['voting', 'stacking']}")
    return constructores[nombre]()


def crear_modelo(nombre: str, params: dict | None = None, pca: int | float | None = None, semilla: int = 42,
                 cache: str | None = None):
    """Crea un Pipeline listo para fit/predict.

    Para "voting" y "stacking", params = {"estimadores": {"svm": {...}, "random_forest": {...}}}.
    `cache`: carpeta donde guardar el escalado y el PCA ya ajustados, para no recalcularlos en cada
    combinación de una búsqueda de hiperparámetros (solo se usa si hay PCA).
    """
    params = dict(params or {})
    if nombre in ("voting", "stacking"):
        estimadores = [(n, _base(n, p or {}, semilla)) for n, p in params.pop("estimadores").items()]
        if nombre == "voting":
            clf = VotingClassifier(estimadores, voting="soft", n_jobs=-1, **params)
        else:
            clf = StackingClassifier(estimadores, final_estimator=LogisticRegression(max_iter=2000),
                                     n_jobs=-1, **params)
    else:
        clf = _base(nombre, params, semilla)
    pasos = [StandardScaler()]
    if pca:
        pasos.append(PCA(n_components=pca, random_state=semilla))
    return make_pipeline(*pasos, clf, memory=cache if pca else None)
