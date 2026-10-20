"""Diccionario clase -> fruta/verdura según el criterio botánico del equipo."""
import csv
import re
from pathlib import Path

ETIQUETAS = ("fruta", "verdura")  # índice 0 = fruta, 1 = verdura
EXCLUIR = "excluir"
MAPPING_PATH = Path(__file__).resolve().parent.parent / "data" / "mapping.csv"


def normalizar(nombre_clase: str) -> str:
    """'Apple Granny Smith 2' -> 'apple granny smith'."""
    nombre = nombre_clase.strip().lower().replace("_", " ")
    return re.sub(r"\s+\d+$", "", nombre)


def cargar_mapping(path: Path = MAPPING_PATH) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {fila["nombre_base"].strip().lower(): fila["etiqueta"].strip() for fila in csv.DictReader(f)}


def etiqueta_de(nombre_clase: str, mapping: dict[str, str]) -> str | None:
    """Etiqueta de la clave más larga que coincide con el inicio del nombre (por palabras completas).

    Devuelve None si la clase no está en el mapeo.
    """
    nombre = normalizar(nombre_clase)
    candidatas = [k for k in mapping if nombre == k or nombre.startswith(k + " ")]
    if not candidatas:
        return None
    return mapping[max(candidatas, key=len)]
