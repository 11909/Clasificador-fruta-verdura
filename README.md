# Clasificador fruta vs verdura

Recibe la foto de un alimento y dice si es **fruta** o **verdura** (criterio botánico). Todo corre en Kaggle; no necesitas instalar nada.

## 1. Prepara tu cuenta (una vez)

1. Crea una cuenta en [kaggle.com](https://www.kaggle.com).
2. Verifica tu teléfono: tu foto → **Settings → Phone Verification**. Sin esto los notebooks no tienen Internet.
3. Pide a Miguel Angel acceso al dataset combinado de Kaggle, el enlace de la carpeta de Google Drive de fotos y el token `GITHUB_TOKEN` (por mensaje privado; nunca lo pegues en el código ni en el chat del grupo).

## 2. Consigue y sube tus fotos (Google Drive)

La carpeta de Drive tiene una subcarpeta por alimento y un `faltantes.txt` que dice qué alimentos te tocan (Persona A, B, C o D) y cuántas fotos faltan de cada uno.

**1. Descarga.** Baja más o menos el doble de lo que pide `faltantes.txt`, porque al limpiar se va la mitad. Con Python:

```
pip install icrawler
```
```python
from icrawler.builtin import BingImageCrawler
for busqueda in ["oca tuberculo", "oca Oxalis tuberosa", "oca tuber market"]:
    BingImageCrawler(storage={"root_dir": "oca"}).crawl(keyword=busqueda, max_num=200, file_idx_offset="auto")
```

Usa varias búsquedas por alimento: en español, en inglés, con el nombre científico y con contexto (`en el mercado`, `cosecha`, `partido`). Si el nombre se confunde con otra cosa, agrega una palabra que lo aclare. Por ejemplo, usa `nabo turnip` y `chile pepper vegetable`.

**2. Limpia.** Revisa la carpeta en vista de miniaturas y borra:
- lo que no sea el alimento: platillos, dibujos, logos, collages o marcas de agua grandes;
- las fotos con fondo blanco de estudio, porque de esas ya hay muchas en Kaggle;
- las de menos de 200 px de lado;
- los duplicados (`dupeGuru` o `czkawka` los encuentran);
- las que mezclan varios alimentos.

## 3. Entrena el modelo (Kaggle)

**Antes, una sola vez:** que Miguel Angel te haya dado acceso al dataset `mendezrosalesmiguel/frankeinstein` y el token `GITHUB_TOKEN` (paso 1).

1. **Abre el notebook:** [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_train.ipynb). Kaggle crea una copia en tu cuenta con la versión más reciente del repo(pucharle al boton cada que se actualiza el repo). Si ya tenías una copia vieja, abre una nueva desde este botón.
2. **Configura la sesión** en el panel derecho (**Settings**):
   - **Internet: On.** Es obligatorio: el notebook clona el repo y descarga el dataset y las fotos de Drive.
   - **Accelerator: None.** No hace falta GPU, porque los modelos corren en CPU.
   - No hace falta **Add Input**.
3. **Agrega el token:** **Add-ons → Secrets → Add a new secret**, con nombre `GITHUB_TOKEN` y el token como valor. Activa la casilla junto al secreto. Sin token entrena igual, pero no publica el modelo.
4. **Ajusta la celda Parámetros** (opcional). Lo principal:

   | Parámetro | Qué hace | Valor por defecto |
   |---|---|---|
   | `MUESTRA_POR_ETIQUETA` | Versión preliminar: toma N frutas + N verduras de todo lo disponible y las divide 80/20. `None` = usar los splits normales (cuando ya haya fotos del equipo). | `5000` |
   | `FRACCION_TEST_MUESTRA` | Parte que va a test en la versión preliminar. | `0.2` |
   | `MAX_POR_CLASE` / `MAX_POR_CLASE_VERDURA` | Imágenes por fruta / verdura en los splits normales. Más es mejor, pero más lento. | `150` / `180` |
   | `LIMPIEZA` | Quita fotos casi repetidas de Fruits-360 y deja máximo 100 de esa fuente por clase. No cambia el dataset de Kaggle. `None` = sin limpiar. | activada |
   | `FEATURES` | Qué se mide de cada foto: color, color por zonas, forma (HOG), textura (LBP) y silueta. | 5 tipos |
   | `CV` | Cómo se mide el desempeño. `uno_fuera_grupo` es leave-one-out por alimento: deja fuera un alimento a la vez. `grupos` hace 5 vueltas y es más rápido. | `uno_fuera_grupo` |
   | `EXCLUIR_GRUPOS` | `["nueces_semillas"]` saca nueces y semillas. | `[]` |
   | `MODELOS` | Los modelos a comparar y sus parámetros de scikit-learn. Quita los que no quieras para que corra más rápido. | 7 modelos |

5. **Córrelo:** **Save Version → Save & Run All → Save**. Corre aunque cierres la pestaña. Kaggle permite hasta 12 horas por corrida, y con los valores por defecto tarda varias horas (el leave-one-out entrena cada modelo ~140 veces). Para probar rápido, usa **Run All** con un `MUESTRA_POR_ETIQUETA` chico, como `500`, y `CV = {"tipo": "grupos", "folds": 5}`.
6. **Revisa los resultados:** abre la versión guardada (desde tu perfil → **Code**) y ve a la pestaña **Output**. Ahí están:
   - `comparacion_modelos.png`: gráficas con las métricas de cada modelo, el tiempo y la matriz de confusión.
   - `metricas_modelos.png`: gráficas de barras que comparan accuracy, precision, recall y F1-score de cada modelo, en validación cruzada y en test.
   - `matrices_modelos.png`: la matriz de confusión en test de cada modelo.
   - `resultados_cv.csv` / `resultados_test.csv`: las mismas métricas de cada modelo en tabla, también por clase (fruta y verdura).
   - `test_metrics.json`: las métricas en test del modelo elegido.
   - `modelo.joblib`: el modelo entrenado.
   - `faltantes.txt`: las fotos que aún le faltan al equipo.
7. **Publicación:** si el modelo nuevo supera en accuracy balanceada al publicado, la última celda lo sube solo como Release de GitHub, y desde ese momento lo usa el paso 4. Para publicar aunque no mejore, pon `FORZAR = True` en esa celda. Para no publicar, pon `PUBLICAR = False`.

## 4. Clasifica fotos

Abre [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/predecir.ipynb) o [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/11909/Clasificador-fruta-verdura/blob/main/notebooks/predecir.ipynb), pega enlaces a fotos (en Google Drive: "cualquier persona con el enlace") y córrelo. Usa el último modelo publicado.

## Si algo falla

| Síntoma | Solución |
|---|---|
| No aparece **Internet** | Verifica tu teléfono (paso 1). |
| "No se pudo clonar el repo" | Activa **Settings → Internet**. |
| "No encontré las carpetas de imágenes" | Activa **Settings → Internet** y pide a Miguel Angel acceso al dataset combinado. |
| `No module named 'src'` | Kaggle reinició la sesión: corre desde la primera celda. |
| "No encontré el secreto GITHUB_TOKEN" | Agrégalo y activa la casilla (paso 3.3). |
| "AVISO: no pude bajar la carpeta de Drive" | El notebook sigue sin las fotos del equipo. Revisa que la carpeta esté compartida como *Cualquier persona con el enlace*, o espera un rato si Drive dice que hubo demasiados accesos. |

El dataset combinado de Kaggle ya está creado (`mendezrosalesmiguel/frankeinstein`); solo se reconstruye si cambian `data/mapping.csv` o las fuentes de Kaggle (ver [docs/detalles.md](docs/detalles.md)).

Todo lo demás (cómo funciona el modelo, el criterio de etiquetas, las fuentes del dataset, uso local) está en [docs/detalles.md](docs/detalles.md).
