from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.mapping import ETIQUETAS


def metricas(reales, predichos) -> dict:
    """Accuracy, F1 por clase y macro, y matriz de confusión (filas = real, columnas = predicho)."""
    etiquetas = list(range(len(ETIQUETAS)))
    f1 = f1_score(reales, predichos, average=None, labels=etiquetas, zero_division=0)
    return {
        "accuracy": float(accuracy_score(reales, predichos)),
        "f1_macro": float(f1.mean()),
        **{f"f1_{e}": float(v) for e, v in zip(ETIQUETAS, f1)},
        "matriz_confusion": confusion_matrix(reales, predichos, labels=etiquetas).tolist(),
    }
