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
