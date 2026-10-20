# Clasificador fruta vs verdura

Clasificador binario de imágenes: recibe la foto de un alimento y responde si es **fruta** o **verdura** (criterio botánico).

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_train.ipynb)

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
  build_dataset.py  # combina varios datasets en uno propio, sin duplicados
configs/baseline.yaml       # rutas, características, tipo de validación y modelos a comparar
data/mapping.csv            # diccionario del criterio botánico (lo revisa el equipo)
notebooks/kaggle_train.ipynb
notebooks/kaggle_build_dataset.ipynb
app/app.py                  # demo con Gradio
scripts/check_mapping.py    # lista clases sin etiqueta en un dataset
configs/dataset.yaml        # fuentes del dataset combinado
tests/
```

## Entrenar en Kaggle

1. Da clic en [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_train.ipynb). Se abre una copia del notebook en tu cuenta de Kaggle, tomada de la versión actual de `main`. Cada quien trabaja en su copia; los cambios de código van al repo, no al notebook.
2. Adjunta el dataset combinado del equipo (o, mientras no exista, **Fruits-360** `moltean/fruits`) y activa **Settings → Internet**. No hace falta GPU.
3. Ejecuta todo, o usa **Save Version → Save & Run All** para que corra sin la pestaña abierta.

## Dataset combinado del equipo

`src/build_dataset.py` junta varias fuentes en un solo dataset con estructura `train/<clase>/` y `test/<clase>/`. Las fuentes se describen en `configs/dataset.yaml`; el notebook [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_build_dataset.ipynb) lo construye en Kaggle y explica cómo publicarlo como dataset privado.

- **Adaptadores por fuente:** cada fuente indica dónde están sus splits, en qué nivel de carpeta está la clase, qué prefijos quitar (`freshapples` → `apple`) y renombres especiales. Los plurales se resuelven solos.
- **Duplicados:** se eliminan por hash del contenido. Si una foto está en test de una fuente y en train de otra, se queda solo en test.
- **Trazabilidad:** `manifest.csv` guarda la fuente y la ruta original de cada imagen, y `reporte.json` los conteos, duplicados y carpetas sin mapeo por fuente.
- **Clases sin mapeo:** no se copian y se listan al final. Se agregan a `data/mapping.csv` y se vuelve a correr.
- **Fuente nueva:** `python -m src.build_dataset --explorar <carpeta>` muestra su estructura para llenar la config.

El dataset combinado mezcla licencias (Fruits-360 es CC BY-SA 4.0): manténganlo privado o revisen las licencias antes de publicarlo.

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
