import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from src.mapping import ETIQUETAS


def metricas(reales, predichos) -> dict:
    """Accuracy, accuracy balanceada (promedio del acierto en fruta y en verdura), precision, recall y F1
    por clase y macro, y matriz de confusión (filas = real, columnas = predicho)."""
    etiquetas = list(range(len(ETIQUETAS)))
    precision, recall, f1, _ = precision_recall_fscore_support(reales, predichos, labels=etiquetas, zero_division=0)
    return {
        "accuracy": float(accuracy_score(reales, predichos)),
        "balanced_accuracy": float(balanced_accuracy_score(reales, predichos)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        **{f"precision_{e}": float(v) for e, v in zip(ETIQUETAS, precision)},
        **{f"recall_{e}": float(v) for e, v in zip(ETIQUETAS, recall)},
        **{f"f1_{e}": float(v) for e, v in zip(ETIQUETAS, f1)},
        "matriz_confusion": confusion_matrix(reales, predichos, labels=etiquetas).tolist(),
    }


def predecir_con_umbral(probabilidades, umbral: float):
    """Etiqueta 1 (verdura) si su probabilidad llega al umbral; con 0.5 es la predicción normal."""
    return (np.asarray(probabilidades)[:, 1] >= umbral).astype(int)


def elegir_umbral(reales, probabilidades) -> float:
    """Umbral de probabilidad de verdura que maximiza la accuracy balanceada. Corrige modelos que,
    por haber visto muchas más frutas, casi nunca pasan de 0.5 en verdura."""
    candidatos = np.round(np.arange(0.05, 0.951, 0.01), 2)
    puntajes = [balanced_accuracy_score(reales, predecir_con_umbral(probabilidades, u)) for u in candidatos]
    # Entre empates se queda con el más cercano a 0.5
    mejor = max(puntajes)
    return float(min((u for u, s in zip(candidatos, puntajes) if s >= mejor - 1e-9), key=lambda u: abs(u - 0.5)))


def intervalos_bootstrap(reales, predichos, grupos, repeticiones: int = 1000, semilla: int = 42,
                         nivel: float = 0.95) -> dict:
    """Intervalo de confianza de accuracy, accuracy balanceada, precision, recall y F1 (macro) en test,
    remuestreando alimentos completos con reemplazo: así refleja qué tanto cambiaría el resultado con
    otros alimentos, no solo con otras fotos de los mismos."""
    reales, predichos, grupos = map(np.asarray, (reales, predichos, grupos))
    nombres = np.unique(grupos)
    # Matriz de confusión 2x2 de cada alimento; un remuestreo es la suma de las de los alimentos elegidos
    por_grupo = np.zeros((len(nombres), 2, 2))
    np.add.at(por_grupo, (np.searchsorted(nombres, grupos), reales, predichos), 1)
    rng = np.random.default_rng(semilla)
    m = por_grupo[rng.integers(0, len(nombres), (repeticiones, len(nombres)))].sum(axis=1)
    aciertos = m[:, [0, 1], [0, 1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.nan_to_num(aciertos / m.sum(axis=2))
        precision = np.nan_to_num(aciertos / m.sum(axis=1))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    valores = {"accuracy": aciertos.sum(1) / m.sum((1, 2)), "balanced_accuracy": recall.mean(1),
               "precision_macro": precision.mean(1), "recall_macro": recall.mean(1), "f1_macro": f1.mean(1)}
    cola = (1 - nivel) / 2 * 100
    return {f"{k}_ic_{lado}": float(np.percentile(v, q)) for k, v in valores.items()
            for lado, q in (("bajo", cola), ("alto", 100 - cola))}
