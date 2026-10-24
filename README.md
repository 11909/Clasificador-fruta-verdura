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

**3. Renombra.** Cada archivo empieza con tu nombre: `ana_001.jpg`. Sirven `.jpg`, `.jpeg`, `.png`, `.webp` y `.bmp`. En Windows, selecciónalas todas → F2 → escribe `ana_` y quedan `ana_ (1).jpg`, `ana_ (2).jpg`...

**4. Sube.** Entra a la carpeta del alimento en Drive (`oca/`, `flor_de_calabaza/`...) y arrastra tus fotos ahí. Si son más de 50, comprímelas en zips de 100 a 200 y sube los zips a esa misma carpeta. No borres archivos de otros.

Si un alimento no tiene carpeta, créala con el nombre de la columna `alimento` de [`data/mapping.csv`](data/mapping.csv), sin acentos y con `_` en vez de espacios. Si no está en el CSV, avisa para agregarlo. No hace falta reconstruir nada: el entrenamiento baja Drive solo.

## 3. Entrena

1. Abre [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/11909/Clasificador-fruta-verdura/blob/main/notebooks/kaggle_train.ipynb).
2. Activa **Settings → Internet**. No hace falta Add Input: el dataset combinado y las fotos de Drive se bajan solos.
3. **Add-ons → Secrets** → agrega `GITHUB_TOKEN` con el token y activa la casilla.
4. En la celda **Parámetros** puedes cambiar los modelos y sus valores. `MUESTRA_POR_ETIQUETA` (5000 por defecto) entrena una versión preliminar con el mismo número de frutas y verduras, divididas 80/20. Ponlo en `None` para usar los splits normales cuando ya haya fotos del equipo.
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
