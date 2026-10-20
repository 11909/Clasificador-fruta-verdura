import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.mapping import ETIQUETAS


@torch.no_grad()
def evaluar(modelo, dataloader, device) -> dict:
    """Accuracy, F1 por clase y macro, y matriz de confusión (filas = real, columnas = predicho)."""
    modelo.eval()
    reales, predichos = [], []
    for x, y in dataloader:
        predichos += modelo(x.to(device)).argmax(1).cpu().tolist()
        reales += y.tolist()
    f1 = f1_score(reales, predichos, average=None, labels=list(range(len(ETIQUETAS))), zero_division=0)
    return {
        "accuracy": accuracy_score(reales, predichos),
        "f1_macro": float(f1.mean()),
        **{f"f1_{e}": float(v) for e, v in zip(ETIQUETAS, f1)},
        "matriz_confusion": confusion_matrix(reales, predichos, labels=list(range(len(ETIQUETAS)))).tolist(),
    }
