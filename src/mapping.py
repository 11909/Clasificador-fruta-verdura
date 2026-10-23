"""Diccionario clase -> fruta/verdura según el criterio botánico del equipo.

data/mapping.csv tiene una fila por alimento con su nombre en español (`alimento`) y los nombres
con que aparece en los datasets (`nombres_dataset`, separados por "|"). Cualquiera de los dos sirve
como nombre de carpeta.
"""
import csv
import re
import unicodedata
from pathlib import Path

ETIQUETAS = ("fruta", "verdura")  # índice 0 = fruta, 1 = verdura
EXCLUIR = "excluir"
MAPPING_PATH = Path(__file__).resolve().parent.parent / "data" / "mapping.csv"


def normalizar(nombre_clase: str) -> str:
    """'Apple Granny Smith 2' -> 'apple granny smith'; 'Limón_amarillo' -> 'limon amarillo'."""
    nombre = unicodedata.normalize("NFKD", nombre_clase.strip().lower().replace("_", " "))
    nombre = "".join(c for c in nombre if not unicodedata.combining(c))
    return re.sub(r"\s+\d+$", "", re.sub(r"\s+", " ", nombre))


def leer_filas(path: Path = MAPPING_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nombres_de(fila: dict) -> list[str]:
    """Nombre en español y nombres en los datasets de una fila, normalizados."""
    otros = [n for n in (fila.get("nombres_dataset") or "").split("|") if n.strip()]
    return [normalizar(n) for n in [fila["alimento"], *otros]]


def cargar_mapping(path: Path = MAPPING_PATH, excluir_grupos=()) -> dict[str, str]:
    """Lee data/mapping.csv como {nombre: etiqueta}, con el nombre en español y los de los datasets.
    Las filas cuyo `grupo` esté en `excluir_grupos` se marcan como "excluir", lo que permite quitar
    un grupo completo (p. ej. "nueces_semillas") sin editar el CSV."""
    excluir_grupos = set(excluir_grupos or ())
    mapping = {}
    for fila in leer_filas(path):
        etiqueta = EXCLUIR if (fila.get("grupo") or "").strip() in excluir_grupos else fila["etiqueta"].strip()
        for nombre in nombres_de(fila):
            mapping[nombre] = etiqueta
    return mapping


def cargar_alimentos(path: Path = MAPPING_PATH) -> dict[str, str]:
    """{nombre: alimento en español}, para que cada clase quede con su nombre en español."""
    return {nombre: normalizar(fila["alimento"]) for fila in leer_filas(path) for nombre in nombres_de(fila)}


def clave_de(nombre_clase: str, mapping: dict) -> str | None:
    """Clave más larga del mapeo que coincide con el inicio del nombre (por palabras completas)."""
    nombre = normalizar(nombre_clase)
    candidatas = [k for k in mapping if nombre == k or nombre.startswith(k + " ")]
    return max(candidatas, key=len) if candidatas else None


def etiqueta_de(nombre_clase: str, mapping: dict[str, str]) -> str | None:
    """Etiqueta de la clase; None si no está en el mapeo."""
    clave = clave_de(nombre_clase, mapping)
    return None if clave is None else mapping[clave]
