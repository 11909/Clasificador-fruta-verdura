"""Módulo de entrada: convierte lo que el usuario da (ruta, enlace, imagen ya abierta o bytes)
en una imagen lista para el modelo.

Las entradas pueden estar alojadas en cualquier sitio con enlace público: el propio repo,
GitHub raw, Google Drive (enlace de compartir), etc. También se acepta un archivo .txt
(local o por enlace) con una entrada por línea, para clasificar varias de una vez.
"""
import io
import re
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from src.dataset import EXTENSIONES

MAX_BYTES = 20 * 1024 * 1024  # no descargar imágenes de más de 20 MB


def normalizar_url(url: str) -> str:
    """Convierte enlaces de "ver" en enlaces de descarga directa."""
    # Google Drive: https://drive.google.com/file/d/<ID>/view?usp=sharing
    drive = re.match(r"https://drive\.google\.com/file/d/([^/]+)", url) or \
        re.match(r"https://drive\.google\.com/open\?id=([^&]+)", url)
    if drive:
        return f"https://drive.google.com/uc?export=download&id={drive.group(1)}"
    # GitHub: https://github.com/<usuario>/<repo>/blob/<rama>/<ruta>
    github = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if github:
        return f"https://raw.githubusercontent.com/{github.group(1)}/{github.group(2)}/{github.group(3)}"
    return url


def descargar_bytes(url: str, timeout: int = 30) -> bytes:
    peticion = urllib.request.Request(normalizar_url(url), headers={"User-Agent": "clasificador-fruta-verdura"})
    with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
        datos = respuesta.read(MAX_BYTES + 1)
    if len(datos) > MAX_BYTES:
        raise ValueError(f"La imagen de {url} pesa más de {MAX_BYTES // 2**20} MB")
    return datos


def leer_imagen(fuente) -> Image.Image:
    """Acepta una ruta, un enlace http(s), bytes, un arreglo de numpy o una imagen de PIL."""
    if isinstance(fuente, Image.Image):
        imagen = fuente
    elif isinstance(fuente, np.ndarray):
        imagen = Image.fromarray(fuente)
    elif isinstance(fuente, (bytes, bytearray)):
        imagen = Image.open(io.BytesIO(fuente))
    elif isinstance(fuente, str) and fuente.startswith(("http://", "https://")):
        imagen = Image.open(io.BytesIO(descargar_bytes(fuente)))
    else:
        imagen = Image.open(Path(fuente))
    imagen.load()
    return imagen.convert("RGB")


def listar_entradas(fuente) -> list:
    """Expande una fuente a la lista de entradas a clasificar:
    - un archivo .txt (local o enlace): una ruta o enlace por línea; las líneas que empiezan con '#' se ignoran
    - una carpeta local: todas sus imágenes (incluye subcarpetas)
    - cualquier otra cosa: una sola entrada
    """
    if isinstance(fuente, (list, tuple)):
        return [e for f in fuente for e in listar_entradas(f)]
    if isinstance(fuente, (str, Path)):
        texto = str(fuente)
        if texto.split("?")[0].lower().endswith(".txt"):
            if texto.startswith(("http://", "https://")):
                contenido = descargar_bytes(texto).decode("utf-8")
            else:
                contenido = Path(texto).read_text(encoding="utf-8")
            lineas = (l.strip() for l in contenido.splitlines())
            return [l for l in lineas if l and not l.startswith("#")]
        if not texto.startswith(("http://", "https://")) and Path(texto).is_dir():
            return sorted(str(p) for p in Path(texto).rglob("*") if p.suffix.lower() in EXTENSIONES)
    return [fuente]


def nombre(fuente) -> str:
    """Texto corto para mostrar de qué entrada salió cada resultado."""
    if isinstance(fuente, (str, Path)):
        return str(fuente)
    return type(fuente).__name__
