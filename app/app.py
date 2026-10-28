"""Interfaz gráfica con Gradio: sube una foto o pega un enlace.

Uso local:      python -m app.app                       (modelo publicado)
                python -m app.app --modelo outputs/modelo.joblib
En un notebook: from app.app import crear_demo; crear_demo().launch()  (en Colab se ve inline; en Kaggle usar share=True)

La interfaz solo llama a Clasificador.clasificar(); para otra interfaz (web, escritorio, móvil)
basta con hacer lo mismo.
"""
import argparse

import gradio as gr

from src.clasificador import Clasificador


def crear_demo(clasificador: Clasificador | None = None) -> gr.Blocks:
    clf = clasificador or Clasificador()

    def clasificar(imagen, enlace):
        entrada = imagen if imagen is not None else (enlace or "").strip()
        if entrada is None or entrada == "":
            raise gr.Error("Sube una foto o pega un enlace.")
        return clf.clasificar(entrada)["probabilidades"]

    with gr.Blocks(title="¿Fruta o verdura?") as demo:
        gr.Markdown(f"# ¿Fruta o verdura?\nCriterio botánico. Modelo: {clf.descripcion}")
        with gr.Row():
            with gr.Column():
                imagen = gr.Image(type="pil", label="Foto")
                enlace = gr.Textbox(label="…o enlace a una imagen (GitHub, Google Drive, etc.)")
                boton = gr.Button("Clasificar", variant="primary")
            salida = gr.Label(label="Predicción")
        boton.click(clasificar, inputs=[imagen, enlace], outputs=salida)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", help="ruta o enlace del modelo (por defecto, el publicado)")
    crear_demo(Clasificador(parser.parse_args().modelo)).launch()
