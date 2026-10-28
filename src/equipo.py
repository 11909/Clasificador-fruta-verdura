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
from src.mapping import cargar_alimentos, cargar_mapping, clave_de, etiqueta_de

PLAN_PATH = Path(__file__).resolve().parent.parent / "data" / "plan_equipo.csv"
# Frutas que solo vienen de Fruits-360: sus fotos reales salen de scripts/web_scrapping.py
PLAN_WEB_PATH = PLAN_PATH.with_name("plan_web.csv")
WEB = "web"


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
        if not zipfile.is_zipfile(zip_path):
            # Una descarga cortada o una página de error de Drive guardada como .zip
            print(f"AVISO: {zip_path.relative_to(carpeta)} no es un zip válido; lo salto. Bórralo y vuelve a correr.")
            continue
        with zipfile.ZipFile(zip_path) as z:
            if zip_path.parent == carpeta:
                destino = destino_raiz / zip_path.stem
                for miembro in _miembros(z):
                    z.extract(miembro, destino)
                raices.append(str(destino))
            else:
                for miembro in _miembros(z):
                    if Path(miembro).suffix.lower() in EXTENSIONES:
                        _soltar(z.read(miembro), zip_path.parent, Path(miembro).name, zip_path.stem)
    return raices


def _soltar(datos: bytes, carpeta: Path, nombre: str, prefijo: str):
    """Escribe una foto de un zip en la carpeta del alimento. Si ya hay otra distinta con ese nombre
    (otro zip de la misma carpeta, o dos subcarpetas del zip con 001.jpg), le antepone el nombre del zip
    y, si hace falta, un número, para no pisarla. Volver a descomprimir deja los mismos nombres."""
    destino, n = carpeta / nombre, 1
    while destino.exists() and destino.read_bytes() != datos:
        # las fotos web_ conservan el prefijo para que sigan contando como fotos de internet
        web, resto = ("web_", nombre[4:]) if nombre.startswith("web_") else ("", nombre)
        destino = carpeta / (f"{web}{prefijo}_{resto}" if n == 1 else f"{web}{prefijo}_{n}_{resto}")
        n += 1
    destino.write_bytes(datos)


def descargar_drive(url: str, destino: str = "/kaggle/working/fotos_equipo", rondas: int = 4,
                    pausa: float = 60) -> list[str]:
    """Baja la carpeta de Drive `url` (compartida como "cualquier persona con el enlace") y
    devuelve las raíces con carpetas de alimentos: la propia carpeta y una por zip de la raíz.

    Primero lista la carpeta y después baja archivo por archivo. Tras muchas descargas seguidas Drive
    empieza a rechazarlas ("demasiados accesos"), y todo lo que va después en orden alfabético se
    quedaba sin bajar; por eso los que fallan se reintentan hasta `rondas` veces, esperando cada vez
    más (`pausa`, 2·`pausa`...). Si aun así faltan, sigue con lo que sí bajó y dice qué alimentos
    quedaron sin fotos. Los archivos que ya estaban de una corrida previa no se vuelven a bajar."""
    import time

    import gdown
    from gdown.exceptions import DownloadError

    archivos = gdown.download_folder(url, output=destino, quiet=True, skip_download=True)
    Path(destino).mkdir(parents=True, exist_ok=True)
    pendientes = [a for a in archivos if not Path(a.local_path).is_file()]
    for ronda in range(rondas):
        if ronda:
            espera = pausa * 2 ** (ronda - 1)
            print(f"Drive rechazó {len(pendientes)} archivos; espero {espera:.0f} s y los reintento "
                  f"(intento {ronda + 1} de {rondas}).")
            time.sleep(espera)
        fallidos = []
        for a in pendientes:
            Path(a.local_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                gdown.download(id=a.id, output=a.local_path, quiet=True, resume=True)
            except DownloadError:
                fallidos.append(a)
        pendientes = fallidos
        if not pendientes:
            break
    if pendientes:
        carpetas = sorted({Path(a.path).parts[0] for a in pendientes})
        print(f"AVISO: {len(pendientes)} archivos de Drive no se pudieron bajar; sigo con los que sí bajaron. "
              f"Carpetas afectadas: {', '.join(carpetas)}. Vuelve a correr esta celda para reintentar "
              "solo los que faltan.")
    for sub in sorted(p for p in Path(destino).iterdir() if p.is_dir()):
        n = sum(1 for p in sub.iterdir() if p.is_file())
        if n == 50:
            print(f"Aviso: {sub.name}/ trae justo 50 archivos; Drive podría haber listado solo 50. "
                  "Si hay más, súbanlos en zips dentro de la carpeta.")
    raices = [destino, *descomprimir_zips(destino)]
    vacias = [sub.name for sub in sorted(p for p in Path(destino).iterdir() if p.is_dir())
              if not any(f.suffix.lower() in EXTENSIONES for f in sub.rglob("*"))]
    if vacias:
        print(f"AVISO: {len(vacias)} carpetas de Drive quedaron sin fotos: {', '.join(vacias)}")
    fotos = sum(1 for r in raices for p in Path(r).rglob("*") if p.suffix.lower() in EXTENSIONES)
    print(f"Drive: {fotos} fotos bajadas")
    return raices


def leer_plan(path: Path = PLAN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def leer_plan_web(path: Path = PLAN_WEB_PATH) -> list[dict]:
    """Frutas del plan de fotos de internet, con persona "web" para que faltantes las cuente aparte. Los
    alimentos del plan del equipo (prioridad "equipo") ya salen en su persona y no se repiten."""
    return [{**fila, "persona": WEB} for fila in leer_plan(path) if fila["prioridad"] != "equipo"]


def contar_por_alimento(muestras: list) -> Counter:
    """{alimento en español: imágenes} a partir de (ruta, etiqueta, clase)."""
    alimentos = cargar_alimentos()
    conteo = Counter()
    for _, _, clase in muestras:
        clave = clave_de(clase, alimentos)
        conteo[alimentos[clave] if clave else clase] += 1
    return conteo


def faltantes(kaggle: Counter, equipo: Counter, max_por_clase: int, fraccion_test: float,
              minimo_equipo: int, plan: list[dict], max_por_clase_verdura: int | None = None) -> list[dict]:
    """Fotos que faltan por alimento del plan para que entrene con `max_por_clase` imágenes
    (`max_por_clase_verdura` en las verduras): las de Kaggle cuentan hasta ese tope y las del equipo
    solo en su parte de train. Cada alimento pide además al menos `minimo_equipo` fotos reales del
    equipo, para que haya fotos fuera de estudio."""
    filas = []
    alimentos, mapping = cargar_alimentos(), cargar_mapping()
    for p in plan:
        alimento = alimentos.get(clave_de(p["alimento"], alimentos), p["alimento"])
        k, e = kaggle.get(alimento, 0), equipo.get(alimento, 0)
        objetivo = (max_por_clase_verdura or max_por_clase) if etiqueta_de(p["alimento"], mapping) == "verdura" \
            else max_por_clase
        necesarias = max(math.ceil(max(0, objetivo - min(k, objetivo)) / (1 - fraccion_test)), minimo_equipo)
        filas.append({"persona": p["persona"], "alimento": p["alimento"], "kaggle": k, "equipo": e,
                      "faltan": max(0, necesarias - e)})
    return filas


def escribir_faltantes(filas: list[dict], destino: Path) -> str:
    """Texto por persona, listo para mandar al grupo; también lo guarda en `destino`."""
    lineas = ["Fotos que faltan por alimento (nombre de carpeta en Drive: faltan).", ""]
    for persona in sorted({f["persona"] for f in filas}):
        propias = [f for f in filas if f["persona"] == persona]
        titulo = "Fotos de internet, scripts/web_scrapping.py" if persona == WEB else f"Persona {persona}"
        lineas.append(f"{titulo} (faltan {sum(f['faltan'] for f in propias)}):")
        lineas += [f"  {f['alimento']}: {f['faltan']}" + ("  (completo)" if not f["faltan"] else "")
                   for f in propias]
        lineas.append("")
    texto = "\n".join(lineas)
    Path(destino).write_text(texto, encoding="utf-8")
    return texto
