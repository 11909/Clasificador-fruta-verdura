# Clasificador fruta vs verdura

Recibe la foto de un alimento y dice si es **fruta** o **verdura** (criterio botánico). Todo corre en Kaggle; no necesitas instalar nada.

## 1. Prepara tu cuenta (una vez)

1. Crea una cuenta en [kaggle.com](https://www.kaggle.com).
2. Verifica tu teléfono: tu foto → **Settings → Phone Verification**. Sin esto los notebooks no tienen Internet.
3. Pide a Miguel Angel acceso de edición al dataset `fotos-equipo` y el token `GITHUB_TOKEN` (por mensaje privado; nunca lo pegues en el código ni en el chat del grupo).

## 2. Sube tus fotos

1. Pon tus fotos en una carpeta por alimento, con su nombre en español y sin acentos: `zanahoria/`, `rabano/`, `limon_amarillo/`. Los nombres válidos son la columna `alimento` de [`data/mapping.csv`](data/mapping.csv). Si falta uno, avisa para agregarlo.
2. Empieza cada archivo con tu nombre: `ana_001.jpg`.
3. Comprime las carpetas en un zip. En Kaggle abre `fotos-equipo` → **New Version**, sube el zip y guarda. Avisen en el grupo para no subir dos a la vez.

## 3. Construye el dataset (una persona, cuando haya cambios)

1. Abre [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_build_dataset.ipynb).
2. Activa **Settings → Internet**. No hace falta Add Input: el notebook descarga los datasets solo.
3. **Save Version → Save & Run All**.
4. En la versión guardada, **Output → New Dataset** la primera vez (privado y compartido con el equipo) o **New Version** las siguientes.

## 4. Entrena

1. Abre [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_train.ipynb).
2. **Add Input** → el dataset del paso 3. Activa **Settings → Internet**.
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
