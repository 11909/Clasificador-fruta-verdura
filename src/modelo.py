"""Módulo del modelo: guarda el modelo entrenado y lo carga desde un archivo o un enlace,
para predecir sin volver a entrenar.

El modelo "oficial" se publica como archivo `modelo.joblib` en un Release de GitHub del repo,
así que cualquiera lo descarga con un enlace fijo (ver docs/detalles.md, sección "Publicar el modelo").
"""
import hashlib
import os
import shutil
import tempfile
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

# Siempre apunta al Release más reciente que tenga un archivo llamado modelo.joblib
MODELO_URL = "https://github.com/11909/Clasificador-fruta-verdura/releases/latest/download/modelo.joblib"
CACHE = Path(os.environ.get("CLASIFICADOR_CACHE", Path.home() / ".cache" / "clasificador-fruta-verdura"))


def guardar_modelo(modelo, features: dict, spec: dict, ruta: Path, umbral: float = 0.5) -> Path:
    """Guarda el modelo junto con lo necesario para usarlo después: la configuración de
    características, el umbral de verdura y la versión de scikit-learn con la que se entrenó."""
    paquete = {
        "modelo": modelo,
        "umbral": umbral,
        "features": features,
        "spec": spec,
        "sklearn": sklearn.__version__,
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, ruta)
    return ruta


def es_url(fuente) -> bool:
    return isinstance(fuente, str) and fuente.startswith(("http://", "https://"))


def descargar(url: str, destino: Path, timeout: int = 60) -> Path:
    """Descarga a un archivo temporal y lo mueve al final, para no dejar archivos a medias."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    peticion = urllib.request.Request(url, headers={"User-Agent": "clasificador-fruta-verdura"})
    with urllib.request.urlopen(peticion, timeout=timeout) as respuesta, \
            tempfile.NamedTemporaryFile(dir=destino.parent, delete=False) as tmp:
        shutil.copyfileobj(respuesta, tmp)
    os.replace(tmp.name, destino)
    return destino


def cargar_modelo(fuente: str | os.PathLike | None = None) -> dict:
    """Carga el modelo desde una ruta local o un enlace.

    Sin argumento usa la variable de entorno MODELO o, si no existe, el Release más reciente.
    Los enlaces se descargan en cada llamada (así siempre se usa la última versión); si la
    descarga falla, se usa la copia guardada de la vez anterior.

    Ojo: un .joblib puede ejecutar código al abrirse. Carguen solo modelos del propio equipo.
    """
    fuente = fuente or os.environ.get("MODELO") or MODELO_URL
    if es_url(fuente):
        copia = CACHE / f"modelo_{hashlib.md5(fuente.encode()).hexdigest()[:10]}.joblib"
        try:
            descargar(fuente, copia)
        except OSError as error:
            if not copia.exists():
                raise RuntimeError(f"No se pudo descargar el modelo de {fuente}: {error}. "
                                   "¿Ya se publicó un Release con modelo.joblib?") from error
            warnings.warn(f"No se pudo descargar el modelo ({error}); uso la copia guardada {copia}.")
        ruta = copia
    else:
        ruta = Path(fuente)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el modelo {ruta}")

    paquete = joblib.load(ruta)
    version = paquete.get("sklearn")
    if version and version != sklearn.__version__:
        warnings.warn(f"El modelo se entrenó con scikit-learn {version} y aquí hay {sklearn.__version__}. "
                      f"Si falla o da resultados raros, instala esa versión: pip install scikit-learn=={version}")
    return paquete


def describir(paquete: dict) -> str:
    return (f"{paquete['spec']['name']} (umbral de verdura {paquete.get('umbral', 0.5)}, scikit-learn {paquete.get('sklearn', '?')}, "
            f"entrenado {paquete.get('creado', 'fecha desconocida')})")
