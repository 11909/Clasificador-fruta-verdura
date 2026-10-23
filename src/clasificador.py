"""Punto único para clasificar imágenes: une el módulo del modelo y el de entrada.

Lo usan la línea de comandos (src/predict.py), los notebooks y la interfaz gráfica (app/app.py),
así que cualquier interfaz nueva solo necesita llamar a Clasificador.clasificar().

    from src.clasificador import Clasificador
    clf = Clasificador()                      # descarga el modelo publicado
    clf.clasificar("https://.../foto.jpg")    # {'etiqueta': 'fruta', 'confianza': 0.93, ...}
"""
from src.entrada import leer_imagen, listar_entradas, nombre
from src.features import extraer
from src.mapping import ETIQUETAS
from src.modelo import cargar_modelo, describir


class Clasificador:
    def __init__(self, modelo=None):
        """modelo: ruta, enlace, o un paquete ya cargado. None = modelo publicado (ver src/modelo.py)."""
        self.paquete = modelo if isinstance(modelo, dict) else cargar_modelo(modelo)

    @property
    def descripcion(self) -> str:
        return describir(self.paquete)

    def probabilidades(self, imagen) -> dict[str, float]:
        x = extraer(leer_imagen(imagen), **self.paquete["features"]).reshape(1, -1)
        modelo = self.paquete["modelo"]
        if hasattr(modelo, "predict_proba"):
            return {e: float(p) for e, p in zip(ETIQUETAS, modelo.predict_proba(x)[0])}
        return {e: float(i == modelo.predict(x)[0]) for i, e in enumerate(ETIQUETAS)}

    def clasificar(self, entrada) -> dict:
        """Clasifica una imagen (ruta, enlace, bytes o imagen de PIL)."""
        probs = self.probabilidades(entrada)
        # El umbral se ajusta al entrenar (ver src/train.py); los modelos viejos no lo traen y usan 0.5
        etiqueta = ETIQUETAS[1] if probs[ETIQUETAS[1]] >= self.paquete.get("umbral", 0.5) else ETIQUETAS[0]
        return {"entrada": nombre(entrada), "etiqueta": etiqueta, "confianza": probs[etiqueta],
                "probabilidades": probs}

    def clasificar_lote(self, fuente) -> list[dict]:
        """Clasifica varias entradas: lista, carpeta o archivo .txt con una entrada por línea.
        Una entrada que falla no detiene las demás; su resultado trae el campo "error"."""
        resultados = []
        for entrada in listar_entradas(fuente):
            try:
                resultados.append(self.clasificar(entrada))
            except Exception as error:  # noqa: BLE001 - se reporta y se sigue con las demás
                resultados.append({"entrada": nombre(entrada), "error": str(error)})
        return resultados
