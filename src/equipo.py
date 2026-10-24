"""Fotos del equipo desde una carpeta compartida de Google Drive.

Cada integrante sube a la carpeta un zip con una subcarpeta por alimento (`zanahoria/ana_001.jpg`).
`descargar_drive` baja los zips con gdown y descomprime cada uno en su propia carpeta, lista para
`dataset.dividir_equipo`.
"""
import zipfile
from pathlib import Path


def descomprimir_zips(carpeta: str) -> list[str]:
    """Descomprime cada .zip de `carpeta` (y subcarpetas) en una carpeta hermana con su nombre.
    Ignora __MACOSX y archivos ocultos que agregan Mac y Windows al comprimir."""
    salidas = []
    for zip_path in sorted(Path(carpeta).rglob("*.zip")):
        destino = zip_path.with_suffix("")
        with zipfile.ZipFile(zip_path) as z:
            for miembro in z.namelist():
                partes = Path(miembro).parts
                if not any(p == "__MACOSX" or p.startswith(".") for p in partes):
                    z.extract(miembro, destino)
        salidas.append(str(destino))
    return salidas


def descargar_drive(url: str, destino: str = "/kaggle/working/fotos_equipo") -> list[str]:
    """Baja los zips de la carpeta de Drive `url` (compartida como "cualquier persona con el
    enlace") y devuelve una carpeta descomprimida por zip."""
    import gdown

    gdown.download_folder(url, output=destino, quiet=True)
    return descomprimir_zips(destino)
