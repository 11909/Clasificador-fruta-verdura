# Clasificador fruta vs verdura

Clasificador binario de imágenes: recibe la foto de un alimento y responde si es **fruta** o **verdura** (criterio botánico).

## Cómo trabajamos

Nada pesado corre en las laptops:

| Qué | Dónde |
|---|---|
| Código | Este repositorio (ramas + Pull Requests) |
| Datos | Kaggle Datasets (Fruits-360 y, más adelante, el dataset combinado del equipo) |
| Entrenamiento | Kaggle Notebooks con GPU (`notebooks/kaggle_train.ipynb`) |
| Experimentos | Weights & Biases (opcional) |
| Modelos | `best.pt` como salida del notebook; el mejor se publica en Hugging Face Hub |
| Demo | Gradio (`app/app.py`), alojable en Hugging Face Spaces |

Las imágenes y los pesos **no se suben al repo** (ver `.gitignore`).

## Estructura

```
src/
  mapping.py    # clase -> fruta/verdura/excluir, a partir de data/mapping.csv
  dataset.py    # lectura de carpetas raiz/clase/imagen, división y transformaciones
  model.py      # modelo de timm preentrenado con cabeza de 2 salidas
  train.py      # entrenamiento (transfer learning) y evaluación final en test
  evaluate.py   # accuracy, F1 por clase y matriz de confusión
  predict.py    # predicción sobre una imagen
configs/baseline.yaml       # hiperparámetros y rutas
data/mapping.csv            # diccionario del criterio botánico (lo revisa el equipo)
notebooks/kaggle_train.ipynb
app/app.py                  # demo con Gradio
scripts/check_mapping.py    # lista clases sin etiqueta en un dataset
tests/
```

## Entrenar en Kaggle

1. Crea un notebook nuevo en Kaggle e importa `notebooks/kaggle_train.ipynb` (File → Import Notebook → GitHub o subir el archivo).
2. Adjunta el dataset **Fruits-360** (`moltean/fruits`), activa GPU e Internet.
3. Ejecuta todo, o usa **Save Version → Save & Run All** para que corra sin la pestaña abierta.

## Criterio de etiquetas

`data/mapping.csv` asigna cada alimento a `fruta`, `verdura` o `excluir`. La coincidencia es por el inicio del nombre de la carpeta, así que `apple` cubre `Apple Granny Smith 2`.

- **Fruta**: todo lo que proviene de la flor y contiene semillas, incluidos tomate, pepino, pimiento, berenjena, calabacita, aguacate y elote.
- **Verdura**: raíces, tubérculos, bulbos, tallos, hojas y flores (zanahoria, papa, camote, cebolla, ajo, jengibre, apio, col, coliflor).
- **Excluir**: hongos, y por ahora nueces y semillas sueltas (almendra, nuez, pistache, cacahuate, castaña). Están marcadas con `revisar` para que el equipo decida.

Si un dataset tiene una clase que no está en el CSV, el entrenamiento se detiene y la lista. `python scripts/check_mapping.py <carpeta>` lo revisa antes.

## Uso local (opcional)

```bash
pip install -r requirements.txt
pytest
python -m src.train --config configs/baseline.yaml   # ajusta las rutas de data/
python -m src.predict outputs/best.pt foto.jpg
python -m app.app outputs/best.pt
```

## Cuidados

- **Fuga de datos:** Fruits-360 tiene muchas fotos casi idénticas de la misma pieza. Usa su partición Test original para la evaluación final; la validación aleatoria dentro de Training sale optimista.
- **Fondo blanco:** con solo Fruits-360 el modelo puede aprender el fondo. Prueben con fotos reales tomadas por el equipo.
- **Desbalance:** hay más clases de fruta que de verdura; la pérdida usa pesos por clase y se reporta F1, no solo accuracy.
- **Secretos:** las API keys van en Kaggle Secrets, nunca en el código.
