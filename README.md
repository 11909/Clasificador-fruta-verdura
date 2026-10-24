# Clasificador fruta vs verdura

Recibe la foto de un alimento y dice si es **fruta** o **verdura** (criterio botánico). Todo corre en Kaggle; no necesitas instalar nada.

## 1. Prepara tu cuenta (una vez)

1. Crea una cuenta en [kaggle.com](https://www.kaggle.com).
2. Verifica tu teléfono: tu foto → **Settings → Phone Verification**. Sin esto los notebooks no tienen Internet.
3. Pide a Miguel Angel acceso al dataset combinado de Kaggle, el enlace de la carpeta de Google Drive de fotos y el token `GITHUB_TOKEN` (por mensaje privado; nunca lo pegues en el código ni en el chat del grupo).

## 2. Sube tus fotos (Google Drive)

La carpeta de Drive ya tiene una subcarpeta por alimento y un `faltantes.txt` con cuántas fotos faltan de cada uno y quién las busca.

1. Entra a la carpeta del alimento (`zanahoria/`, `flor_de_calabaza/`...) y sube ahí tus fotos.
2. Empieza cada archivo con tu nombre: `ana_001.jpg`. Sirven `.jpg`, `.jpeg`, `.png`, `.webp` y `.bmp`.
3. Si son más de 50 fotos de un alimento, súbelas en zips de 100 a 200 dentro de la misma carpeta. No borres archivos de otros.

Para un alimento que no tiene carpeta, créala con el nombre de la columna `alimento` de [`data/mapping.csv`](data/mapping.csv) (sin acentos y con `_` en vez de espacios). Si no está en el CSV, avisa para agregarlo. No hace falta reconstruir nada: el entrenamiento baja Drive solo y manda ~30 % de las fotos a test.

## 3. Entrena

1. Abre [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_train.ipynb).
2. Activa **Settings → Internet**. No hace falta Add Input: el dataset combinado y las fotos de Drive se bajan solos.
3. **Add-ons → Secrets** → agrega `GITHUB_TOKEN` con el token y activa la casilla.
4. Si quieres, cambia los modelos y sus parámetros en la celda **Parámetros**.
5. **Save Version → Save & Run All**. En **Output** queda `faltantes.txt` con las fotos que aún faltan. Si el modelo mejora al anterior, se publica solo en GitHub.

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
| "No pude bajar la carpeta de Drive" | Revisa que la carpeta esté compartida como *Cualquier persona con el enlace*, o espera un rato si Drive dice que hubo demasiados accesos. |

El dataset combinado de Kaggle ya está creado (`mendezrosalesmiguel/frankeinstein`); solo se reconstruye si cambian `data/mapping.csv` o las fuentes de Kaggle (ver [docs/detalles.md](docs/detalles.md)).

Todo lo demás (cómo funciona el modelo, el criterio de etiquetas, las fuentes del dataset, uso local) está en [docs/detalles.md](docs/detalles.md).
