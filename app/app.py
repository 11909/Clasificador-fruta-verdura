"""Demo con Gradio.  Uso local:  python -m app.app outputs/modelo.joblib

En Hugging Face Spaces: sube modelo.joblib junto a este archivo y define MODELO=modelo.joblib.
"""
import os
import sys

import gradio as gr

from src.predict import cargar, predecir

ruta = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODELO", "outputs/modelo.joblib")
paquete = cargar(ruta)

demo = gr.Interface(
    fn=lambda imagen: predecir(paquete, imagen),
    inputs=gr.Image(type="pil", label="Foto de una fruta o verdura"),
    outputs=gr.Label(label="Predicción"),
    title="¿Fruta o verdura?",
    description=f"Clasificación con criterio botánico. Modelo: {paquete['spec']['name']}.",
)

if __name__ == "__main__":
    demo.launch()
