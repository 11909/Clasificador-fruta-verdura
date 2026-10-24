# Clasificador fruta vs verdura

Recibe la foto de un alimento y dice si es **fruta** o **verdura** (criterio botánico). Todo corre en Kaggle; no necesitas instalar nada.

## 1. Prepara tu cuenta (una vez)

1. Crea una cuenta en [kaggle.com](https://www.kaggle.com).
2. Verifica tu teléfono: tu foto → **Settings → Phone Verification**. Sin esto los notebooks no tienen Internet.
3. Pide a Miguel Angel el enlace de la carpeta de Google Drive de fotos y el token `GITHUB_TOKEN` (por mensaje privado; nunca lo pegues en el código ni en el chat del grupo).

## 2. Construye el dataset (una persona; la primera vez o si cambia `data/mapping.csv`)

1. Abre [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_build_dataset.ipynb).
2. Activa **Settings → Internet**. No hace falta Add Input: el notebook descarga los datasets solo.
3. **Save Version → Save & Run All**.
4. En la versión guardada, **Output → New Dataset** la primera vez (privado; en **Settings → Sharing** agrega al equipo con *Can edit*) o **New Version** las siguientes.

## 3. Sube tus fotos (Google Drive)

1. Pon una carpeta por alimento con su nombre en español y sin acentos: `zanahoria/`, `rabano/`, `flor_de_calabaza/`. Los nombres válidos son la columna `alimento` de [`data/mapping.csv`](data/mapping.csv). Si falta uno, avisa para agregarlo.
2. Empieza cada archivo con tu nombre: `ana_001.jpg`.
3. Comprime las carpetas en un zip con tu nombre y la fecha (`ana_2026-10-24.zip`) y súbelo a la carpeta compartida de Drive. No borres ni reemplaces zips de otros.

La primera vez, Miguel Angel crea la carpeta en Drive, la comparte con el equipo como *Editor* y además en **Compartir → Cualquier persona con el enlace → Lector** (así el notebook la puede bajar), y pega el enlace en `equipo_drive` de [`configs/baseline.yaml`](configs/baseline.yaml). No hace falta reconstruir nada: el entrenamiento baja los zips solo y manda ~30 % de las fotos a test.

## 4. Entrena

1. Abre [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_train.ipynb).
2. **Add Input** → el dataset combinado del paso 2 (las fotos de Drive se bajan solas). Activa **Settings → Internet**.
3. **Add-ons → Secrets** → agrega `GITHUB_TOKEN` con el token y activa la casilla.
4. **Save Version → Save & Run All**. Si el modelo mejora al anterior, se publica solo en GitHub.

## 5. Clasifica fotos

Abre [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/predecir.ipynb) o [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/11909/Clasificador-fruta-verdura/blob/main/notebooks/predecir.ipynb), pega enlaces a fotos (en Google Drive: "cualquier persona con el enlace") y córrelo. Usa el último modelo publicado.

## Si algo falla

| Síntoma | Solución |
|---|---|
| No aparece **Internet** | Verifica tu teléfono (paso 1). |
| "No se pudo clonar el repo" | Activa **Settings → Internet**. |
| "No encontré las carpetas de imágenes" | Falta adjuntar el dataset en **Add Input**. |
| `No module named 'src'` | Kaggle reinició la sesión: corre desde la primera celda. |
| "No encontré el secreto GITHUB_TOKEN" | Agrégalo y activa la casilla (paso 4.3). |

Todo lo demás (cómo funciona el modelo, el criterio de etiquetas, las fuentes del dataset, uso local) está en [docs/detalles.md](docs/detalles.md).
