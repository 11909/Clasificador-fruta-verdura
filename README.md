# Clasificador fruta vs verdura

Clasificador binario de imágenes: recibe la foto de un alimento y responde si es **fruta** o **verdura** (criterio botánico). Solo usa modelos clásicos de aprendizaje automático; **no usa redes neuronales**.

## Inicio rápido

| Quiero… | Notebook |
|---|---|
| Entrenar y comparar modelos | `kaggle_train.ipynb` [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_train.ipynb) |
| Clasificar imágenes con el modelo ya entrenado (sin volver a entrenar) | `predecir.ipynb` [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/predecir.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/11909/Clasificador-fruta-verdura/blob/main/notebooks/predecir.ipynb) |
| Construir el dataset combinado del equipo | `kaggle_build_dataset.ipynb` [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_build_dataset.ipynb) |

El ciclo completo es: **entrenar** en Kaggle → **publicar** `modelo.joblib` en un Release de GitHub → **clasificar** desde cualquier lugar con enlaces a imágenes.

## Cómo trabajamos

El flujo es: imagen → vector de 388 características (color HSV, forma con HOG y textura con LBP) → modelo de scikit-learn. Todo corre en CPU.

Nada pesado corre en las laptops:

| Qué | Dónde |
|---|---|
| Código | Este repositorio (ramas + Pull Requests) |
| Datos | Kaggle Datasets: Fruits-360 y, cuando exista, el dataset combinado del equipo |
| Entrenamiento | Kaggle Notebooks en CPU |
| Experimentos | `resultados_cv.csv` y `test_metrics.json` de cada corrida |
| Modelo publicado | `modelo.joblib` en un Release de GitHub; se descarga con un enlace fijo |
| Imágenes a clasificar | Cualquier enlace público (GitHub, Google Drive, enlace directo) |
| Interfaz | Gradio (`app/app.py`), en el notebook o local |

Las imágenes y los modelos **no se suben al repo** (ver `.gitignore`); los modelos van en Releases.

## Estructura

```
src/
  mapping.py        # clase -> fruta/verdura/excluir, a partir de data/mapping.csv
  dataset.py        # lectura de carpetas raiz/clase/imagen y muestreo por clase
  features.py       # extracción de características: color (HSV), forma (HOG), textura (LBP)
  models.py         # catálogo de modelos de scikit-learn, incluidos voting y stacking
  train.py          # validación cruzada de todos los modelos, reentrena el mejor y evalúa en test
  evaluate.py       # accuracy, F1 por clase y matriz de confusión
  modelo.py         # módulo del modelo: guardar y cargar (de archivo o del Release publicado)
  publicar.py       # publica el modelo entrenado como Release de GitHub
  entrada.py        # módulo de entrada: rutas, enlaces (GitHub, Drive), .txt con enlaces, carpetas
  clasificador.py   # une modelo y entrada; lo usan la terminal, los notebooks y la interfaz
  predict.py        # clasificar desde la terminal
  build_dataset.py  # combina varios datasets en uno propio, sin duplicados
app/app.py          # interfaz gráfica con Gradio (foto o enlace)
configs/
  baseline.yaml     # rutas, características, tipo de validación y modelos a comparar
  dataset.yaml      # fuentes del dataset combinado
data/mapping.csv    # diccionario del criterio botánico (lo revisa el equipo)
entradas/ejemplos.txt  # enlaces de ejemplo para clasificar
notebooks/          # kaggle_train, predecir y kaggle_build_dataset
scripts/check_mapping.py  # lista clases sin etiqueta en un dataset
tests/
```

## Entrenar en Kaggle

1. Abre [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_train.ipynb). Se crea una copia del notebook en tu cuenta, tomada de la versión actual de `main`. Los cambios de código van al repo, no al notebook.
2. Adjunta el dataset con **Add Input → Datasets**: el combinado del equipo o, mientras no exista, **Fruits-360** de `moltean`. Ojo: es un *dataset*, no un notebook ni un modelo.
3. Activa **Settings → Internet** (Kaggle pide verificar tu teléfono). No hace falta GPU.
4. Usa **Save Version → Save & Run All** para que corra sin la pestaña abierta y las salidas queden guardadas.

Al terminar, en `outputs/` quedan `resultados_cv.csv` (comparación de modelos, con los segundos que tardó cada uno), `test_metrics.json` (calificación final en Test) y `modelo.joblib` (el modelo elegido). La última celda lo prueba con `entradas/ejemplos.txt`; si te convence, [publícalo](#publicar-el-modelo).

## Predecir con el modelo ya entrenado (sin volver a entrenar)

La predicción está separada en módulos, así que el modelo se entrena una vez y se reutiliza:

| Módulo | Qué hace |
|---|---|
| `src/modelo.py` | Guarda el modelo y lo carga desde un archivo o un enlace. Por defecto descarga el `modelo.joblib` del **Release más reciente** del repo. |
| `src/publicar.py` | Sube el modelo entrenado como Release nuevo; lo llama la última celda del notebook de entrenamiento. |
| `src/entrada.py` | Convierte la entrada en imagen: ruta, enlace (GitHub, Google Drive, cualquier enlace directo), `.txt` con un enlace por línea, o carpeta. |
| `src/clasificador.py` | `Clasificador().clasificar(entrada)` devuelve etiqueta, confianza y probabilidades. Es lo único que necesita una interfaz. |
| `src/predict.py`, `app/app.py`, `notebooks/predecir.ipynb` | Interfaces: terminal, interfaz gráfica (Gradio) y notebook. Todas llaman al mismo `Clasificador`. |

```bash
python -m src.predict entradas/ejemplos.txt                 # modelo publicado + enlaces de ejemplo
python -m src.predict https://.../foto.jpg mi_foto.jpg
python -m app.app                                           # interfaz gráfica en el navegador
```

```python
from src.clasificador import Clasificador
clf = Clasificador()                                        # se descarga una vez
clf.clasificar("https://drive.google.com/file/d/<ID>/view") # {'etiqueta': 'fruta', 'confianza': 0.93, ...}
```

### Publicar el modelo

Es automático: al final de `notebooks/kaggle_train.ipynb` hay una celda que crea un Release nuevo (etiqueta `modelo-AAAAMMDD-HHMMSS`) con `modelo.joblib`, `test_metrics.json` y `resultados_cv.csv`. Desde ese momento, todo lo de arriba usa ese modelo; los Releases anteriores quedan como historial. Basta con **Run All** (o **Save Version → Save & Run All**).

- **Solo publica si mejora:** compara la accuracy balanceada de test con la del Release más reciente. Si no la supera, no publica y lo dice. Con `FORZAR = True` en esa celda publica de todas formas, por ejemplo al cambiar de dataset de prueba (las métricas dejan de ser comparables). Con `PUBLICAR = False` solo entrena.
- **Sin token no publica**, pero el notebook termina bien.

**Token del equipo.** Hay un solo token, creado por el dueño del repo y compartido por mensaje privado. Nadie más tiene que crear uno.

- **Agregarlo (cada integrante, una vez):** en el notebook de Kaggle, **Add-ons → Secrets → Add a new secret**, con nombre `GITHUB_TOKEN` y el token como valor. Activen la casilla para que el notebook lo use.
- **Cómo se creó (solo el dueño):** GitHub → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**. En *Repository access* solo `Clasificador-fruta-verdura`, en *Permissions* **Contents: Read and write**, y con fecha de vencimiento. Al vencer, se genera otro y se comparte igual.
- **Cuidados:** el token permite subir código al repo, así que es solo para el equipo. Nunca va en el código, el notebook ni el chat del grupo; solo en Kaggle Secrets, que no se copian al compartir el notebook. Si se filtra, el dueño lo revoca en GitHub y comparte uno nuevo.

Para publicar desde otra máquina: `GITHUB_TOKEN=... python -m src.publicar --salida outputs`.

Si hace falta publicar a mano: **Releases → Draft a new release**, etiqueta nueva, adjunta `modelo.joblib` (con ese nombre exacto) y `test_metrics.json`.

El modelo guarda la versión de scikit-learn con la que se entrenó; si al cargarlo hay otra, se muestra un aviso con el comando para instalar la correcta. Solo carguen modelos del propio equipo: un `.joblib` puede ejecutar código al abrirse.

### Fotos de prueba

Pongan sus fotos en cualquier sitio con enlace público y agreguen los enlaces a un `.txt` (como `entradas/ejemplos.txt`). En Google Drive, compártanlas como "cualquier persona con el enlace"; el enlace normal de compartir sirve tal cual.

## Dataset combinado del equipo

`src/build_dataset.py` junta varias fuentes en un solo dataset con estructura `train/<clase>/` y `test/<clase>/`. Las fuentes se describen en `configs/dataset.yaml`; el notebook `kaggle_build_dataset.ipynb` lo construye en Kaggle y explica cómo publicarlo como dataset privado.

Fuentes actuales: Fruits-360, Fresh and Rotten, PackEat, Vegetable Image Dataset (15 clases), vegetales de Bangladesh (Mendeley, CC BY 4.0), Grocery Store Dataset (GitHub, MIT) y las fotos del equipo. Las rutas marcadas "por confirmar" en la config se revisan con `--explorar`.

**`data/mapping.csv` es la lista oficial de alimentos.** Tiene una fila por alimento con su nombre en español (`alimento`), su etiqueta y los nombres con que aparece en los datasets (`nombres_dataset`, separados por `|`; p. ej. `berenjena` ← `eggplant|aubergine|brinjal`). En el dataset combinado cada clase queda con su nombre en español, sin acentos (`limon amarillo`, `rabano`).

**Fotos del equipo:** van en un dataset privado de Kaggle con una carpeta por alimento, con su nombre en español y sin carpetas de train/test ni de fruta/verdura:

```
fotos-equipo/
  oca/
  rabano/
  camote/
  ...
```

- `python -m src.build_dataset --plantilla fotos-equipo` crea una carpeta vacía por cada alimento del CSV (menos los excluidos).
- Las carpetas pueden llevar acentos o guiones bajos (`limón_amarillo` = `limon amarillo`), y lo que sigue al nombre se ignora (`rabano_negro` cuenta como `rabano`).
- Para un alimento que no está, primero se agrega su fila a `data/mapping.csv` y luego su carpeta.
- El script decide si cada foto va a train o a test (`fraccion_test`, 30 % por defecto) a partir de su nombre, así que agregar fotos no mueve las que ya estaban.

### Crear el dataset por primera vez

Lo hace una sola persona (la dueña de los datasets en Kaggle):

1. **Fotos del equipo.** En una computadora con el repo: `python -m src.build_dataset --plantilla fotos-equipo`, pon al menos una foto en las carpetas que ya tengan (Kaggle no guarda carpetas vacías) y comprime `fotos-equipo` en un zip. En kaggle.com → **Datasets → New Dataset**, sube el zip, nómbralo `fotos-equipo` y déjalo **privado**. En **Settings → Sharing** agrega a cada integrante con permiso **Can edit**.
2. **Vegetales de Bangladesh.** Descarga el zip de [Mendeley](https://data.mendeley.com/datasets/rtx9ngb68j/2) y súbelo igual, como `vegetales-bangladesh`, privado.
3. **Construir.** Abre `kaggle_build_dataset.ipynb` en Kaggle, agrega como input las fuentes que lista su primera celda y activa Internet. Corre la sección 1 y, si alguna ruta de `configs/dataset.yaml` no coincide, corrígela en la celda 2 (y luego en el repo). Después **Save Version → Save & Run All**.
4. **Publicar.** En la versión guardada, **Output → New Dataset** (privado, compartido con el equipo). Ese es el dataset que se adjunta en `kaggle_train.ipynb`.

### Subir o actualizar fotos (cada integrante)

1. Pon tus fotos en carpetas con el nombre del alimento, igual que en la plantilla, y nombra los archivos con tu nombre al inicio (`miguel_001.jpg`) para no pisar fotos de otros.
2. Comprime esas carpetas en un zip. En el dataset `fotos-equipo` → **New Version**, sube el zip (se agrega a lo que ya hay) y guarda. Para quitar o cambiar una foto, bórrala o súbela de nuevo en esa misma pantalla.
3. Avisa al equipo; una persona vuelve a correr `kaggle_build_dataset.ipynb` (**Save & Run All**) y en **Output → New Version** actualiza el dataset combinado. Revisen `conteos.csv` para ver cómo quedó el balance.

Eviten dos versiones al mismo tiempo: si dos personas suben a la vez, la última versión puede no incluir las fotos de la otra.

- **Adaptadores por fuente:** cada fuente indica dónde están sus splits, en qué nivel de carpeta está la clase, qué prefijos quitar (`freshapples` → `manzana`) y renombres especiales. Los plurales se resuelven solos.
- **Duplicados:** se eliminan por hash del contenido. Si una foto está en test de una fuente y en train de otra, se queda solo en test.
- **Trazabilidad:** `manifest.csv` guarda la fuente y la ruta original de cada imagen, y `reporte.json` los conteos, duplicados y carpetas sin mapeo por fuente.
- **Desbalance:** `conteos.csv` tiene una fila por etiqueta, clase y fuente con sus imágenes de train y test, de la clase con más imágenes a la que tiene menos.
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

`data/mapping.csv` asigna cada alimento a `fruta`, `verdura` o `excluir`, y opcionalmente a un `grupo`. La coincidencia es por el inicio del nombre de la carpeta, en español o con alguno de los `nombres_dataset`, así que `apple` cubre `Apple Granny Smith 2`.

- **Fruta**: todo lo que proviene de la flor y contiene semillas, incluidos tomate, pepino, pimiento, berenjena, calabacita, aguacate y elote.
- **Verdura**: raíces, tubérculos, bulbos, tallos, hojas y flores (zanahoria, papa, camote, cebolla, ajo, jengibre, apio, col, coliflor).
- **Nueces y semillas** (almendra, nuez, avellana, castaña, pistache, cacahuate, marañón): por ahora cuentan como **fruta**. Están en el grupo `nueces_semillas` del CSV; para sacarlas del modelo basta con poner `excluir_grupos: [nueces_semillas]` en la config, sin tocar código ni el CSV.
- **Excluir**: hongos.

Si un dataset tiene una clase que no está en el CSV, el entrenamiento se detiene y la lista. `python scripts/check_mapping.py <carpeta>` lo revisa antes.

## Uso local (opcional)

Clasificar no necesita GPU ni el dataset, así que sí corre bien en una laptop:

```bash
pip install -r requirements.txt
python -m src.predict entradas/ejemplos.txt          # usa el modelo publicado
python -m app.app                                    # interfaz gráfica en http://127.0.0.1:7860
pytest                                               # pruebas
python -m src.train --config configs/baseline.yaml   # entrenar local: ajusta las rutas de data/
```

## Problemas comunes en Kaggle

| Síntoma | Causa y solución |
|---|---|
| No aparece la opción **Internet** | La cuenta no tiene el teléfono verificado: tu foto → **Settings → Phone Verification**. |
| "No se pudo clonar el repo" | Internet está apagado en el notebook: **Settings → Internet**. |
| "No encontré las carpetas de imágenes" | No hay dataset adjunto, o se adjuntó un notebook o un modelo en vez del dataset. |
| `No module named 'src'` o falta `/kaggle/working/repo` | La sesión se reinició y Kaggle borró `/kaggle/working`. Corre desde la primera celda. |
| Aviso de pip `dependency conflicts` (p. ej. `dopamine-rl`, `gym`) | Paquetes preinstalados de Kaggle que no usamos. Se puede ignorar. |
| "¿Ya se publicó un Release con modelo.joblib?" | Todavía no hay modelo publicado; ver [Publicar el modelo](#publicar-el-modelo). |
| "No encontré el secreto GITHUB_TOKEN" | Falta agregar o activar el secreto; ver [Publicar el modelo](#publicar-el-modelo). |
| "GitHub rechazó el token" | El token venció o no tiene permiso de escritura; genera otro. |
| Aviso de versión de scikit-learn | El modelo se entrenó con otra versión; instala la que indica el aviso. |
| Una copia vieja del notebook falla | Abre una copia nueva desde los botones de [Inicio rápido](#inicio-rápido). |

## Cuidados

- **Fuga de datos:** Fruits-360 tiene muchas fotos casi idénticas de la misma pieza. Por eso la validación cruzada agrupa por variedad y la evaluación final usa la partición Test original.
- **Fondo blanco:** con solo Fruits-360 el modelo puede aprender el fondo. Prueben con fotos reales tomadas por el equipo.
- **Desbalance:** Fruits-360 tiene ~15 imágenes de fruta por cada una de verdura. Por eso se entrena con el mismo número de ambas (`balancear_etiquetas`), los modelos usan `class_weight="balanced"` donde aplica, se ajusta el umbral de verdura con validación cruzada (`ajustar_umbral`) y se reporta accuracy balanceada y F1, no solo accuracy.
- **Secretos:** las API keys van en Kaggle Secrets, nunca en el código.
