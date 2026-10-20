"""Demo con Gradio.  Uso local:  python -m app.app outputs/best.pt

En Hugging Face Spaces: sube best.pt junto a este archivo y define CHECKPOINT=best.pt.
"""
import os
import sys

import gradio as gr

from src.predict import cargar, predecir

checkpoint = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CHECKPOINT", "outputs/best.pt")
modelo, transform = cargar(checkpoint)

demo = gr.Interface(
    fn=lambda imagen: predecir(modelo, transform, imagen),
    inputs=gr.Image(type="pil", label="Foto de una fruta o verdura"),
    outputs=gr.Label(label="Predicción"),
    title="¿Fruta o verdura?",
    description="Clasificación con criterio botánico.",
)

if __name__ == "__main__":
    demo.launch()
