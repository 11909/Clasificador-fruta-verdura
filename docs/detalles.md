# Clasificador fruta vs verdura: detalles

Referencia completa. Para empezar, sigue la guía corta del [README](../README.md).

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
  features.py       # extracción de características: color (HSV, global y por zonas), forma (HOG), textura (LBP) y silueta
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
scripts/web_scrapping.py  # fotos de Wikimedia Commons para las frutas que solo tienen Fruits-360
tests/
```

## Entrenar en Kaggle

1. Abre [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_train.ipynb). Se crea una copia del notebook en tu cuenta, tomada de la versión actual de `main`. Los cambios de código van al repo, no al notebook.
2. Adjunta el dataset combinado con **Add Input** (`mendezrosalesmiguel/frankeinstein`). En una sesión interactiva el notebook lo puede descargar solo con `kagglehub` si falta, pero en **Save & Run All** Kaggle no deja descargar datasets que no estén adjuntos.
3. Activa **Settings → Internet** (Kaggle pide verificar tu teléfono). No hace falta GPU.
4. Usa **Save Version → Save & Run All** para que corra sin la pestaña abierta y las salidas queden guardadas.

Al terminar, en `outputs/` quedan `resultados_cv.csv` y `resultados_test.csv` (accuracy, precision, recall y F1 de cada modelo en validación cruzada y en test, con los segundos que tardó cada uno), `metricas_modelos.png` y `matrices_modelos.png` (las mismas métricas en gráficas), `test_metrics.json` (calificación final en Test) y `modelo.joblib` (el modelo elegido). La última celda lo prueba con `entradas/ejemplos.txt`; si te convence, [publícalo](#publicar-el-modelo).

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

Es automático: al final de `notebooks/kaggle_train.ipynb` hay una celda que crea un Release nuevo (etiqueta `modelo-AAAAMMDD-HHMMSS`) con `modelo.joblib`, `test_metrics.json`, `resultados_cv.csv`, `resultados_test.csv` y las gráficas. Desde ese momento, todo lo de arriba usa ese modelo; los Releases anteriores quedan como historial. Basta con **Run All** (o **Save Version → Save & Run All**).

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

Por ahora el dataset se arma **solo con datasets públicos de Kaggle**: Fruits-360, Fresh and Rotten, PackEat y Vegetable Image Dataset (15 clases). Las fotos del equipo van aparte, en una carpeta de Google Drive. Las rutas marcadas "por confirmar" en la config se revisan con `--explorar`.

**`data/mapping.csv` es la lista oficial de alimentos.** Tiene una fila por alimento con su nombre en español (`alimento`), su etiqueta y los nombres con que aparece en los datasets (`nombres_dataset`, separados por `|`; p. ej. `berenjena` ← `eggplant|aubergine|brinjal`). En el dataset combinado cada clase queda con su nombre en español, sin acentos (`limon amarillo`, `rabano`).

### Crear el dataset

1. **Construir.** Abre `kaggle_build_dataset.ipynb` en Kaggle, activa Internet (los datasets se descargan solos con `kagglehub`, sin Add Input). La sección 1 muestra la estructura de cada fuente; si alguna ruta de `configs/dataset.yaml` no coincide, corrígela en la celda 2 (y luego en el repo). Después **Save Version → Save & Run All**.
2. **Publicar.** En la versión guardada, **Output → New Dataset** (privado, compartido con el equipo con *Can edit*). Ese es el dataset que se adjunta en `kaggle_train.ipynb`.
3. **Actualizar.** Si cambia `data/mapping.csv` o la config, se vuelve a correr y se usa **Output → New Version** sobre el mismo dataset. Revisen `conteos.csv` para ver el balance.

### Fotos del equipo

Viven en una carpeta compartida de Google Drive, no en Kaggle. (El dataset combinado de Kaggle no sirve para esto porque los datasets creados desde el Output de un notebook no aceptan archivos subidos a mano.)

```
carpeta de Drive/
  faltantes.txt
  zanahoria/ana_001.jpg, ana_002.jpg, luis_lote1.zip ...
  flor_de_calabaza/...
  eva_2026-10-24.zip      (opcional: zip con su propio árbol de alimentos)
```

- `data/plan_equipo.csv` dice qué alimento busca cada persona. Las carpetas de Drive salen de ahí.
- El notebook de entrenamiento baja la carpeta con `gdown` (`EQUIPO_DRIVE` en la celda Parámetros, que debe estar compartida como "cualquier persona con el enlace") y descomprime los zips (`src/equipo.py`). Un zip dentro de una carpeta de alimento suelta sus fotos ahí; un zip en la raíz se lee como un árbol de alimentos aparte. `__MACOSX` y los archivos ocultos se ignoran.
- Cada foto va a test con probabilidad `fraccion_test_equipo` (30 %) según su carpeta y nombre, así que agregar fotos no mueve las que ya estaban.
- **faltantes.txt:** el notebook cuenta las imágenes de train de Kaggle y las del equipo por alimento del plan. Pide las fotos que hacen falta para llegar a `MAX_POR_CLASE` en train (las del equipo cuentan al 70 % porque el resto va a test) y al menos `MINIMO_EQUIPO` fotos reales. Lo escribe en `/kaggle/working/outputs/faltantes.txt`. El `faltantes.txt` inicial de Drive es un estimado hecho antes de tener los conteos reales.
- Las carpetas pueden llevar acentos o guiones bajos (`limón_amarillo` = `limon amarillo`), y lo que sigue al nombre se ignora (`rabano_negro` cuenta como `rabano`). Si una carpeta no está en `data/mapping.csv`, el notebook se detiene y la lista.
- Drive puede listar solo las primeras 50 entradas de una carpeta; si una carpeta trae justo 50, el notebook avisa. Por eso conviene subir lotes grandes como zips.

#### Fotos de internet para las frutas con solo Fruits-360

Casi todas las frutas fuera de `plan_equipo.csv` solo traen fotos de estudio de Fruits-360. `data/plan_web.csv` las lista con su prioridad, cuántas fotos reales necesita cada una y qué buscar en Wikimedia Commons. La meta sale de la misma cuenta que `faltantes.txt`: Fruits-360 aporta hasta 100 fotos de train y de las nuevas el 70 % va a train, así que 80 fotos (100 en prioridad alta) dejan cada fruta en 150 a 170 de train, sin pasar de las 180 de cada verdura. Las que no tienen nada en Kaggle (plátano macho y las calabazas de la fuente de Bangladesh, que está desactivada) piden 215 = 150 / 0.7. Las cuentas salen de `conteos.csv` del dataset combinado; okra no está porque ya trae fotos reales de Fresh and Rotten.

```
pip install pillow numpy icrawler torch transformers
python scripts/web_scrapping.py --excluir ruta/a/fotos_prueba
```

- El script no depende del resto del repo: basta con `web_scrapping.py` y `plan_web.csv` en la misma carpeta.
- `plan_web.csv` también trae los alimentos de `plan_equipo.csv` que aún faltan en Drive (prioridad `equipo`, ~12,900 fotos, ya sin el lote de las Personas A, B y C). `--prioridad equipo` baja solo esos; `--faltantes faltantes.txt` toma las metas del `faltantes.txt` que escribe el notebook en Output, con los conteos reales de Drive.
- Fuentes (`--fuentes`, por defecto `bing,commons`): **Bing** (con icrawler) es la más rápida, con varias descargas a la vez, unos 30 a 60 minutos para todo el plan, pero sin licencia conocida y con más ruido. **Wikimedia Commons** tiene licencias libres, pero limita la velocidad y obliga a bajar de una en una, así que tarda varias horas: conviene solo para completar. Commons completa lo que Bing no alcanzó. Openverse, que junta Flickr y otros sitios, quedó fuera porque sin cuenta solo permite 200 consultas al día. iNaturalist también quedó fuera porque casi todas sus fotos son de la planta, no de la fruta.
- Las búsquedas llevan contexto (`oca tuberculo andino`, `tunas fruta de nopal`) y en Bing palabras a excluir (`-ave`, `-receta` y la columna `excluir` del plan, como `-ganso` en oca o `-pescado` en tuna).
- Si están instalados `torch` y `transformers`, un modelo CLIP revisa cada foto contra "una foto de <alimento>" (columna `nombre_en`) y contra aves, personas, platillos, dibujos, edificios, etc., y descarta lo que no es el alimento. `--umbral` ajusta qué tan estricto es (0.6 por defecto) y `--sin-filtro` lo apaga. `--revisar` lo pasa por las fotos ya bajadas y mueve las que no pasan a `salida_web/descartadas/`.
- Descarta dibujos, plantas, platillos, fotos de estudio con fondo blanco, las de menos de 200 px y duplicados.
- `--excluir` recibe la carpeta de fotos de prueba: ninguna foto que esté ahí (por título de Commons o por hash perceptual) entra a entrenamiento, para que la prueba siga siendo con fotos que el modelo nunca vio.
- Deja `salida_web/fotos/<alimento>/web_<alimento>_0001.jpg` y `salida_web/fuentes_web.csv` (fuente, autor y licencia de cada foto). Revisa `fotos/` en miniaturas y borra lo que no sea el alimento. Lo borrado no se vuelve a bajar; correr el script otra vez completa la meta con fotos nuevas.
- **Subir:** arrastra las carpetas de `salida_web/fotos/` a la raíz de la carpeta de Drive. Si la carpeta de ese alimento ya existe en Drive, sube las fotos dentro de la existente (Drive no junta carpetas con el mismo nombre). Si Drive tarda con tantas fotos sueltas, `--empaquetar` arma un zip por carpeta en `salida_web/para_drive/`, que el notebook también descomprime. Las fotos `web_*` cuentan como fotos de Drive, con la misma división fija 70/30; `FOTOS_WEB = False` las deja fuera para comparar.
- `--nucleos` limita los núcleos del procesador que usa el filtro (2 por defecto) para que la computadora no se caliente tanto.
- **Balance:** al entrenar, cada clase toma primero sus fotos de Drive y se completa con el dataset combinado hasta `MAX_POR_CLASE` (`MAX_POR_CLASE_VERDURA` en verduras). En la versión preliminar (`MUESTRA_POR_ETIQUETA`) cada etiqueta se reparte por igual entre clases, con el mismo orden. Así, más fotos en Drive no agrandan una clase sobre las demás.
- El `faltantes.txt` del notebook termina con una sección "Fotos de internet" con lo que aún falta de cada fruta.

#### Renombrar y subir las fotos

**Renombrar.** Cada archivo empieza con tu nombre: `ana_001.jpg`. Sirven `.jpg`, `.jpeg`, `.png`, `.webp` y `.bmp`. En Windows, selecciónalas todas → F2 → escribe `ana_` y quedan `ana_ (1).jpg`, `ana_ (2).jpg`...

**Subir.** Entra a la carpeta del alimento en Drive (`oca/`, `flor_de_calabaza/`...) y arrastra tus fotos ahí. Si son más de 50, comprímelas en zips de 100 a 200 y sube los zips a esa misma carpeta. No borres archivos de otros.

Si un alimento no tiene carpeta, créala con el nombre de la columna `alimento` de [`data/mapping.csv`](../data/mapping.csv), sin acentos y con `_` en vez de espacios. Si no está en el CSV, avisa para agregarlo. No hace falta reconstruir nada: el entrenamiento baja Drive solo.

### Más adelante: fuentes fuera de Kaggle

`configs/dataset.yaml` ya trae, comentadas, dos fuentes más. Para activarlas se quita el `# ` de sus líneas:

- **Vegetales de Bangladesh** (Mendeley, CC BY 4.0): se descarga el zip de [Mendeley](https://data.mendeley.com/datasets/rtx9ngb68j/2) y se sube a Kaggle como dataset privado `vegetales-bangladesh`.
- **Grocery Store Dataset** (GitHub, MIT): en el notebook, antes de construir, se clona con `!git clone -q --depth 1 https://github.com/marcusklasson/GroceryStoreDataset.git /kaggle/working/fuentes/GroceryStoreDataset`.
### Cómo combina las fuentes

- **Adaptadores por fuente:** cada fuente indica dónde están sus splits, en qué nivel de carpeta está la clase, qué prefijos quitar (`freshapples` → `manzana`) y renombres especiales. Los plurales se resuelven solos.
- **Duplicados:** se eliminan por hash del contenido. Si una foto está en test de una fuente y en train de otra, se queda solo en test.
- **Trazabilidad:** `manifest.csv` guarda la fuente y la ruta original de cada imagen, y `reporte.json` los conteos, duplicados y carpetas sin mapeo por fuente.
- **Desbalance:** `conteos.csv` tiene una fila por etiqueta, clase y fuente con sus imágenes de train y test, de la clase con más imágenes a la que tiene menos.
- **Clases sin mapeo:** no se copian y se listan al final. Se agregan a `data/mapping.csv` y se vuelve a correr.
- **Fuente nueva:** `python -m src.build_dataset --explorar <carpeta>` muestra su estructura para llenar la config.

El dataset combinado mezcla licencias (Fruits-360 es CC BY-SA 4.0): manténganlo privado o revisen las licencias antes de publicarlo.

## Modelos y validación

En `configs/baseline.yaml` se listan los modelos a comparar. Cada uno se evalúa con la validación cruzada elegida (ver abajo), y uno se reentrena con todo el conjunto de entrenamiento y se evalúa en Test. Con `elegir_por: cv` (por defecto) se elige por la validación cruzada, que mide alimentos que no vio. Con `elegir_por: vistos` se elige el de mayor F1 macro con el 20 % de las fotos de cada alimento apartadas de train, es decir, con fotos nuevas de alimentos que el modelo sí vio, como al usar la app. En una corrida con el test del dataset combinado, en una corrida eso eligió naive_bayes (0.82 en test) aunque svm sacó 0.955.

- **Modelos disponibles:** `logistic_regression`, `svm`, `knn`, `naive_bayes`, `decision_tree`, `random_forest`, `gradient_boosting`, y las combinaciones `voting` y `stacking`.
- **Validación:**
  - `anidada` (por defecto): validación cruzada anidada. Las vueltas externas (`folds`, 5) separan por alimento y sirven para comparar los modelos; en cada una, los modelos con `grid` eligen sus hiperparámetros con otras vueltas internas por alimento (`folds_internos`, 3) usando solo la parte de entrenamiento de esa vuelta. Así la métrica externa no está inflada por la búsqueda. Se reporta promedio ± desviación de cada métrica entre vueltas, y los hiperparámetros elegidos en cada vuelta. El modelo final repite la búsqueda con todo el entrenamiento.
  - `uno_fuera_grupo` usa `LeaveOneGroupOut`: cada vuelta deja fuera un alimento completo y lo predice con un modelo que nunca lo vio. Como cada vuelta tiene un solo alimento, las predicciones de todas las vueltas se juntan y las métricas se calculan una vez (por eso `f1_macro_std` sale 0).
  - `grupos` usa `StratifiedGroupKFold` con 5 vueltas, de modo que una misma variedad nunca está a la vez en entrenamiento y validación. Mide lo mismo que la anterior, pero más rápido.
  - `estratificada` usa `StratifiedKFold`. Sale más alto porque valida con alimentos que el modelo ya vio.
  - `uno_fuera` usa `LeaveOneOut`, una imagen por vuelta. Entrena el modelo tantas veces como imágenes haya, así que solo sirve con muestras chicas (`MUESTRA_POR_ETIQUETA` de 300 o menos). Además, fotos casi iguales de la misma pieza quedan en entrenamiento y validación, y el resultado sale inflado.
- **Intervalos en test:** con `bootstrap` (por defecto 1000) se remuestrean alimentos completos del conjunto de prueba y se reporta el intervalo del 95 % de cada métrica (`_ic_bajo`, `_ic_alto` en `resultados_test.csv`).
- **Ajuste de hiperparámetros:** agrega `grid` a un modelo para usar `GridSearchCV` con la misma validación. El notebook trae grids para KNN (`n_neighbors`, `weights`), SVM (`C`, `gamma`) y Random Forest (`max_depth`, `min_samples_leaf`); `ajustar_hiperparametros: false` los ignora. Con `uno_fuera_grupo` o `uno_fuera` la búsqueda usa 5 vueltas por grupos y el mejor modelo se mide después con leave-one-out.
- **Características** (`features.tipos`):
  - `color`: histograma HSV de toda la imagen, más media y desviación.
  - `color_espacial`: media y desviación HSV en una cuadrícula de 3x3, más un histograma tono-saturación del centro, donde suele estar el alimento.
  - `hog` / `hog_fino`: forma y bordes; `hog_fino` usa celdas de 8 px en lugar de 16.
  - `lbp` / `lbp_multi`: textura; `lbp_multi` la mide a 3 escalas (radios 1, 2 y 3).
  - `forma`: separa el alimento del fondo por saturación y mide su área, redondez, alargamiento, solidez y excentricidad.
  - `segmentar` separa el alimento del fondo antes de medir. `saturacion` toma los píxeles más saturados (Otsu); `grabcut` parte de esa máscara y la refina con GrabCut de OpenCV, tomando el borde de la imagen como fondo. Se queda la región más grande; el color se mide solo sobre ella, el fondo se pinta de gris liso antes de HOG y LBP, y `forma` usa esa misma máscara. Si la máscara sale casi vacía o casi completa, se usa la imagen entera.
  - `color_lab`: media y desviación en Lab y un histograma conjunto de los canales a y b, medidos sobre el alimento. Lab separa mejor los tonos que HSV.
  - `bovw` (bolsa de palabras visuales): descriptores SIFT en una rejilla densa (225 por foto, a 128 px), contados contra un vocabulario de 256 patrones locales que se aprende con k-means de las fotos de train y se guarda con el modelo.
  - Por defecto: `color`, `color_espacial`, `forma`, `color_lab` y `bovw` (489 dimensiones). Con 4,051 fotos reales de 40 alimentos del proyecto, validando con fotos nuevas de alimentos vistos, gradient_boosting pasó de 0.73 a 0.78 de accuracy balanceada frente a los tipos anteriores (1,981 dimensiones), y svm de 0.60 a 0.71. Con alimentos no vistos quedó igual (~0.60).
  - Los modelos publicados antes guardan sus propios tipos, así que siguen funcionando.
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
- **Ruido de Fruits-360:** son cuadros de video de una pieza girando sobre fondo blanco, cientos por variedad. Al cargar el dataset, `src/limpieza.py` calcula un hash perceptual (dHash de 64 bits) de cada foto de Fruits-360, quita las que difieren en 6 bits o menos de otra de su clase y deja como máximo 100 por clase (`data.limpieza`). Opcionalmente descarta las fotos de fondo blanco liso de cualquier fuente. El dataset de Kaggle no cambia.
- **Desbalance:** Fruits-360 tiene ~15 imágenes de fruta por cada una de verdura. Por eso se entrena con el mismo número de ambas (`balancear_etiquetas`), los modelos usan `class_weight="balanced"` donde aplica (salvo `gradient_boosting`: con train ya balanceado los pesos no cambian nada y con ellos tarda unas 14 veces más), se ajusta el umbral de verdura con validación cruzada (`ajustar_umbral`) y se reporta accuracy balanceada y F1, no solo accuracy.
- **Secretos:** las API keys van en Kaggle Secrets, nunca en el código.
