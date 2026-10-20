# Clasificador fruta vs verdura

Clasificador binario de imágenes: recibe la foto de un alimento y responde si es **fruta** o **verdura** (criterio botánico).

## Cómo trabajamos

Solo se usan modelos clásicos de aprendizaje automático (regresión logística, SVM, KNN, Naive Bayes, árboles, random forest, gradient boosting y sus combinaciones). **No se usan redes neuronales.**

El flujo es: imagen → vector de características (color, forma con HOG y textura con LBP) → modelo de scikit-learn. Todo corre en CPU; extraer las características de Fruits-360 completo toma unos minutos.

Nada pesado corre en las laptops:

| Qué | Dónde |
|---|---|
| Código | Este repositorio (ramas + Pull Requests) |
| Datos | Kaggle Datasets (Fruits-360 y, más adelante, el dataset combinado del equipo) |
| Entrenamiento | Kaggle Notebooks en CPU (`notebooks/kaggle_train.ipynb`) |
| Experimentos | `resultados_cv.csv` y `test_metrics.json` de cada corrida |
| Modelos | `modelo.joblib` como salida del notebook; el mejor se publica en Hugging Face Hub |
| Demo | Gradio (`app/app.py`), alojable en Hugging Face Spaces |

Las imágenes y los pesos **no se suben al repo** (ver `.gitignore`).

## Estructura

```
src/
  mapping.py    # clase -> fruta/verdura/excluir, a partir de data/mapping.csv
  dataset.py    # lectura de carpetas raiz/clase/imagen y muestreo por clase
  features.py   # extracción de características: color (HSV), forma (HOG), textura (LBP)
  models.py     # catálogo de modelos de scikit-learn, incluidos voting y stacking
  train.py      # validación cruzada de todos los modelos, reentrena el mejor y evalúa en test
  evaluate.py   # accuracy, F1 por clase y matriz de confusión
  predict.py    # predicción sobre una imagen
configs/baseline.yaml       # rutas, características, tipo de validación y modelos a comparar
data/mapping.csv            # diccionario del criterio botánico (lo revisa el equipo)
notebooks/kaggle_train.ipynb
app/app.py                  # demo con Gradio
scripts/check_mapping.py    # lista clases sin etiqueta en un dataset
tests/
```

## Entrenar en Kaggle

1. Crea un notebook nuevo en Kaggle e importa `notebooks/kaggle_train.ipynb` (File → Import Notebook → GitHub o subir el archivo).
2. Adjunta el dataset **Fruits-360** (`moltean/fruits`) y activa Internet. No hace falta GPU.
3. Ejecuta todo, o usa **Save Version → Save & Run All** para que corra sin la pestaña abierta.

## Modelos y validación

En `configs/baseline.yaml` se listan los modelos a comparar. Cada uno se evalúa con validación cruzada de 5 folds, y el de mayor F1 macro se reentrena con todo el conjunto de entrenamiento y se evalúa en Test.

- **Modelos disponibles:** `logistic_regression`, `svm`, `knn`, `naive_bayes`, `decision_tree`, `random_forest`, `gradient_boosting`, y las combinaciones `voting` y `stacking`.
- **Validación:** `cv.tipo: grupos` usa `StratifiedGroupKFold`, de modo que una misma variedad nunca está a la vez en entrenamiento y validación. `estratificada` usa `StratifiedKFold`.
- **Ajuste de hiperparámetros:** agrega `grid` a un modelo para usar `GridSearchCV` con la misma validación.
- **Reducción de dimensiones:** `pca` aplica PCA antes del modelo.

## Criterio de etiquetas

`data/mapping.csv` asigna cada alimento a `fruta`, `verdura` o `excluir`, y opcionalmente a un `grupo`. La coincidencia es por el inicio del nombre de la carpeta, así que `apple` cubre `Apple Granny Smith 2`.

- **Fruta**: todo lo que proviene de la flor y contiene semillas, incluidos tomate, pepino, pimiento, berenjena, calabacita, aguacate y elote.
- **Verdura**: raíces, tubérculos, bulbos, tallos, hojas y flores (zanahoria, papa, camote, cebolla, ajo, jengibre, apio, col, coliflor).
- **Nueces y semillas** (almendra, nuez, avellana, castaña, pistache, cacahuate, marañón): por ahora cuentan como **fruta**. Están en el grupo `nueces_semillas` del CSV; para sacarlas del modelo basta con poner `excluir_grupos: [nueces_semillas]` en la config, sin tocar código ni el CSV.
- **Excluir**: hongos.

Si un dataset tiene una clase que no está en el CSV, el entrenamiento se detiene y la lista. `python scripts/check_mapping.py <carpeta>` lo revisa antes.

## Uso local (opcional)

```bash
pip install -r requirements.txt
pytest
python -m src.train --config configs/baseline.yaml   # ajusta las rutas de data/
python -m src.predict outputs/modelo.joblib foto.jpg
python -m app.app outputs/modelo.joblib
```

## Cuidados

- **Fuga de datos:** Fruits-360 tiene muchas fotos casi idénticas de la misma pieza. Por eso la validación cruzada agrupa por variedad y la evaluación final usa la partición Test original.
- **Fondo blanco:** con solo Fruits-360 el modelo puede aprender el fondo. Prueben con fotos reales tomadas por el equipo.
- **Desbalance:** hay más clases de fruta que de verdura; los modelos usan `class_weight="balanced"` donde aplica y se reporta F1, no solo accuracy.
- **Secretos:** las API keys van en Kaggle Secrets, nunca en el código.
