"""Fotos del equipo desde una carpeta compartida de Google Drive.

La carpeta tiene una subcarpeta por alimento (`zanahoria/`, `flor_de_calabaza/`...). Dentro de cada
una se suben fotos sueltas o zips de fotos. También se acepta un zip en la raíz con una subcarpeta
por alimento adentro. `descargar_drive` baja todo con gdown y descomprime los zips, y
`faltantes` dice cuántas fotos faltan por alimento.
"""
import csv
import math
import zipfile
from collections import Counter
from pathlib import Path

from src.dataset import EXTENSIONES
from src.mapping import cargar_alimentos, clave_de

PLAN_PATH = Path(__file__).resolve().parent.parent / "data" / "plan_equipo.csv"


def _miembros(z: zipfile.ZipFile) -> list[str]:
    """Archivos del zip sin __MACOSX ni ocultos, que agregan Mac y Windows al comprimir."""
    return [m for m in z.namelist() if not m.endswith("/")
            and not any(p == "__MACOSX" or p.startswith(".") for p in Path(m).parts)]


def descomprimir_zips(carpeta: str, destino_raiz: str | None = None) -> list[str]:
    """Descomprime los zips de `carpeta`.

    - Un zip dentro de una carpeta de alimento suelta sus fotos en esa carpeta.
    - Un zip en la raíz trae su propio árbol de alimentos y se descomprime en
      `destino_raiz/<nombre del zip>`. Devuelve esas carpetas.
    """
    carpeta = Path(carpeta)
    destino_raiz = Path(destino_raiz) if destino_raiz else carpeta.parent / f"{carpeta.name}_zips"
    raices = []
    for zip_path in sorted(carpeta.rglob("*.zip")):
        with zipfile.ZipFile(zip_path) as z:
            if zip_path.parent == carpeta:
                destino = destino_raiz / zip_path.stem
                for miembro in _miembros(z):
                    z.extract(miembro, destino)
                raices.append(str(destino))
            else:
                for miembro in _miembros(z):
                    if Path(miembro).suffix.lower() in EXTENSIONES:
                        (zip_path.parent / Path(miembro).name).write_bytes(z.read(miembro))
    return raices


def descargar_drive(url: str, destino: str = "/kaggle/working/fotos_equipo") -> list[str]:
    """Baja la carpeta de Drive `url` (compartida como "cualquier persona con el enlace") y
    devuelve las raíces con carpetas de alimentos: la propia carpeta y una por zip de la raíz."""
    import gdown

    gdown.download_folder(url, output=destino, quiet=True)
    for sub in sorted(p for p in Path(destino).iterdir() if p.is_dir()):
        n = sum(1 for p in sub.iterdir() if p.is_file())
        if n == 50:
            print(f"Aviso: {sub.name}/ trae justo 50 archivos; Drive podría haber listado solo 50. "
                  "Si hay más, súbanlos en zips dentro de la carpeta.")
    return [destino, *descomprimir_zips(destino)]


def leer_plan(path: Path = PLAN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def contar_por_alimento(muestras: list) -> Counter:
    """{alimento en español: imágenes} a partir de (ruta, etiqueta, clase)."""
    alimentos = cargar_alimentos()
    conteo = Counter()
    for _, _, clase in muestras:
        clave = clave_de(clase, alimentos)
        conteo[alimentos[clave] if clave else clase] += 1
    return conteo


def faltantes(kaggle: Counter, equipo: Counter, max_por_clase: int, fraccion_test: float,
              minimo_equipo: int, plan: list[dict]) -> list[dict]:
    """Fotos que faltan por alimento del plan para que entrene con `max_por_clase` imágenes:
    las de Kaggle cuentan completas y las del equipo solo en su parte de train. Cada alimento pide
    además al menos `minimo_equipo` fotos reales del equipo, para que haya fotos fuera de estudio."""
    filas = []
    alimentos = cargar_alimentos()
    for p in plan:
        alimento = alimentos.get(clave_de(p["alimento"], alimentos), p["alimento"])
        k, e = kaggle.get(alimento, 0), equipo.get(alimento, 0)
        necesarias = max(math.ceil(max(0, max_por_clase - k) / (1 - fraccion_test)), minimo_equipo)
        filas.append({"persona": p["persona"], "alimento": p["alimento"], "kaggle": k, "equipo": e,
                      "faltan": max(0, necesarias - e)})
    return filas


def escribir_faltantes(filas: list[dict], destino: Path) -> str:
    """Texto por persona, listo para mandar al grupo; también lo guarda en `destino`."""
    lineas = ["Fotos que faltan por alimento (nombre de carpeta en Drive: faltan).", ""]
    for persona in sorted({f["persona"] for f in filas}):
        propias = [f for f in filas if f["persona"] == persona]
        lineas.append(f"Persona {persona} (faltan {sum(f['faltan'] for f in propias)}):")
        lineas += [f"  {f['alimento']}: {f['faltan']}" + ("  (completo)" if not f["faltan"] else "")
                   for f in propias]
        lineas.append("")
    texto = "\n".join(lineas)
    Path(destino).write_text(texto, encoding="utf-8")
    return texto
