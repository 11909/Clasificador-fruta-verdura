import timm

from src.mapping import ETIQUETAS


def crear_modelo(nombre: str, preentrenado: bool = True):
    """Modelo de timm preentrenado en ImageNet con una cabeza nueva de 2 salidas (fruta, verdura)."""
    return timm.create_model(nombre, pretrained=preentrenado, num_classes=len(ETIQUETAS))


def congelar_backbone(modelo, congelar: bool):
    """Con congelar=True solo se entrena la cabeza clasificadora."""
    cabeza = set(id(p) for p in modelo.get_classifier().parameters())
    for p in modelo.parameters():
        p.requires_grad = (not congelar) or id(p) in cabeza
