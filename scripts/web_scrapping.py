"""Descarga fotos reales de internet para las frutas que solo tienen Fruits-360 y las deja listas para Drive.

Uso (Python 3.10+, `pip install pillow numpy icrawler`):
    python web_scrapping.py --solo rambutan --meta 10           # prueba rápida
    python web_scrapping.py --excluir ruta/a/fotos_prueba       # todo el plan
    python web_scrapping.py --faltantes faltantes.txt           # metas con los conteos reales de Drive
    python web_scrapping.py --empaquetar                        # rehace para_drive/ después de revisar

Lee `plan_web.csv` (junto a este archivo, o `data/plan_web.csv` del repo): alimento, prioridad, meta de
fotos y búsquedas. Trae dos grupos:
    frutas con solo Fruits-360 (prioridad sin_fotos, alta, media, secos): ~6,000 fotos
    alimentos del plan del equipo que aún faltan en Drive (prioridad equipo): ~12,900 fotos, según
    faltantes.txt menos el lote de las Personas A, B y C. Con `--faltantes faltantes.txt` (el que escribe
    el notebook en Output, con los conteos reales de Drive) la meta de cada alimento sale de ahí.
    `--prioridad equipo` baja solo ese grupo; `--prioridad sin_fotos,alta,media,secos`, solo las frutas.
Deja:
    salida_web/fotos/<alimento>/web_<alimento>_0001.jpg   para revisar en miniaturas
    salida_web/fuentes_web.csv                            origen, autor y licencia de cada foto
    salida_web/para_drive/<alimento>/web_<alimento>_1.zip lo que se sube a Drive

Fuentes (`--fuentes`, en orden; la segunda completa lo que no dio la primera):
    bing     Bing Imágenes con icrawler: la más rápida (varias descargas a la vez, ~30 a 60 min para todo
             el plan), pero sin licencia conocida y con más ruido. El dataset debe quedarse privado.
    commons  Wikimedia Commons: licencias libres y menos ruido, pero Wikimedia limita la velocidad y
             obliga a bajar una foto a la vez (varias horas para todo el plan; mejor solo para completar).
Por defecto usa las dos: `--fuentes bing,commons`.

La meta por alimento sale de la misma cuenta que `faltantes.txt`: cada fruta entrena con hasta
MAX_POR_CLASE = 150 fotos, Fruits-360 aporta como máximo 100 tras la limpieza y de las fotos nuevas el
70 % va a train. Así, 80 fotos (100 en prioridad alta) dejan cada fruta en 150 a 170 de train, sin
pasar de las 180 de cada verdura; las que no tienen nada en Kaggle piden 215 = 150 / 0.7. El notebook
además toma primero las fotos de Drive y completa cada clase con el dataset combinado hasta su tope.

Se descartan las fotos con fondo blanco de estudio, las de menos de 200 px, los dibujos y los
duplicados (también entre alimentos). `--excluir` recibe la carpeta de fotos de prueba (fotos_prueba/):
ninguna foto que esté ahí, por título de Commons o por hash perceptual, entra al entrenamiento.
Si se corta o se vuelve a correr, sigue donde se quedó; las fotos que borres al revisar no se vuelven
a bajar.
"""
import argparse
import csv
import io
import json
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
PLAN = next((p for p in (AQUI / "plan_web.csv", AQUI.parent / "data" / "plan_web.csv") if p.exists()),
            AQUI / "plan_web.csv")
UA = "clasificador-fruta-verdura/1.0 (https://github.com/11909/clasificador-fruta-verdura)"
API = "https://commons.wikimedia.org/w/api.php"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANCHO = 500        # tamaño estándar de miniatura de Commons; el dataset guarda las imágenes a 256 px
LADO_MAXIMO = 500  # las fotos se guardan con este lado mayor
LADO_MINIMO = 200
UMBRAL_DUPLICADO = 6
MAX_RESULTADOS = 400  # por búsqueda en Commons; más abajo casi todo es ruido
FOTOS_POR_ZIP = 150   # Drive lista solo 50 archivos sueltos por carpeta, así que se sube en zips
COLUMNAS = ["archivo", "alimento", "prioridad", "fuente", "busqueda", "titulo", "autor", "licencia",
            "pagina_fuente", "url_descarga", "estado"]
# Títulos que casi nunca son la fruta sola: dibujos, láminas, plantas en flor, platillos y bebidas
EXCLUIR = re.compile(
    r"(drawing|illustration|diagram|logo|\bmap\b|icon|painting|stamp|herbari|fruits[-_ ]?360|engraving|"
    r"lithograph|\bplate\b|tafel|k[oö]hler|botanical|\bflowers?\b|blossom|\bflor(es)?\b|\btrees?\b|"
    r"\bleaf\b|leaves|seedling|\bplants?\b|\bjuice|\bjam\b|\bcake|\bpie\b|salad|smoothie|dessert|"
    r"ice[ _-]?cream|cocktail|\bwine\b|liqueur|recipe|\bdish\b|\bcurry\b|stall[ _-]?sign|\bcoins?\b|"
    r"\bbooks?\b|poster|\blabel|\bpdf\b|\btiff?\b|\bsvg\b|clipart|vector|cartoon)", re.I)


# --- Imágenes: mismo hash y misma detección de fondo blanco que src/limpieza.py del repo ---

def dhash(im: Image.Image, lado: int = 8) -> int:
    gris = np.asarray(im.convert("L").resize((lado + 1, lado)), dtype=np.int16)
    bits = (gris[:, 1:] > gris[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def fondo_blanco(im: Image.Image, claro: int = 230, fraccion: float = 0.9) -> bool:
    """True si al menos `fraccion` del borde (5 % de cada lado) es casi blanco: foto de estudio."""
    a = np.asarray(im.convert("RGB").resize((64, 64)))
    borde = np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3), a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)])
    return (borde.min(axis=1) >= claro).mean() >= fraccion


def abrir(datos: bytes) -> Image.Image | None:
    try:
        with Image.open(io.BytesIO(datos)) as im:
            return im.convert("RGB")
    except Exception:
        return None


class Hashes:
    """Hashes perceptuales ya vistos, para no repetir fotos ni meter las de prueba."""

    def __init__(self):
        self.h = np.zeros(0, dtype=np.uint64)

    def agregar(self, h: int):
        self.h = np.append(self.h, np.uint64(h))

    def repetido(self, h: int) -> bool:
        if not len(self.h):
            return False
        bits = np.unpackbits(np.bitwise_xor(self.h, np.uint64(h)).view(np.uint8)).reshape(len(self.h), 64)
        return bits.sum(axis=1).min() <= UMBRAL_DUPLICADO


# --- Fuentes: cada una da (datos de la foto, bytes) ---

def pedir(url: str, intentos: int = 5) -> bytes:
    """GET con User-Agent propio y espera creciente si el servidor pide bajar la velocidad (429/503)."""
    for i in range(intentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == intentos - 1:
                raise
            espera = int(e.headers.get("Retry-After") or 0) or 5 * 2 ** i
        except (urllib.error.URLError, TimeoutError):
            if i == intentos - 1:
                raise
            espera = 5 * 2 ** i
        time.sleep(min(espera, 120))
    raise RuntimeError("sin respuesta")


def candidatos_commons(busqueda: str):
    """Archivos de Commons para `busqueda`, en el orden de relevancia, de 50 en 50 (lo máximo por llamada)."""
    for offset in range(0, MAX_RESULTADOS, 50):
        params = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
                  "gsrsearch": f"filetype:bitmap {busqueda}", "gsrlimit": 50, "gsroffset": offset,
                  "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": ANCHO}
        datos = json.loads(pedir(API + "?" + urllib.parse.urlencode(params)))
        paginas = sorted(datos.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
        for p in paginas:
            info = (p.get("imageinfo") or [{}])[0]
            if info.get("mime") not in ("image/jpeg", "image/png", "image/webp") or EXCLUIR.search(p["title"]):
                continue
            # Si el original es más angosto que ANCHO, Commons da el original, y upload.wikimedia.org
            # rechaza las descargas de originales con 429
            if info.get("width", 0) <= ANCHO or info.get("height", 0) < LADO_MINIMO:
                continue
            meta = info.get("extmetadata", {})
            autor = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
            yield {"fuente": "commons", "titulo": p["title"], "url_descarga": info.get("thumburl") or info["url"],
                   "pagina_fuente": info.get("descriptionurl", ""), "autor": autor[:200],
                   "licencia": meta.get("LicenseShortName", {}).get("value", "")}
        if "continue" not in datos:
            return
        time.sleep(1)


def commons(busqueda: str, vistos: set, pausa: float, **_):
    for c in candidatos_commons(busqueda):
        if c["titulo"] in vistos:
            yield c, None  # ya vista o de prueba: se cuenta como descarte sin bajarla
            continue
        try:
            datos = pedir(c["url_descarga"])
        except Exception as e:
            print(f"  falló {c['url_descarga']}: {e}")
            continue
        finally:
            time.sleep(pausa)
        yield c, datos


def bing(busqueda: str, vistos: set, cuantas: int, hilos: int = 4, **_):
    """Bing Imágenes con icrawler: baja `cuantas` fotos a la vez en una carpeta temporal."""
    try:
        from icrawler import ImageDownloader
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        sys.exit("Para la fuente bing falta icrawler: pip install icrawler (o usa --fuentes commons)")
    urls = {}

    class Registro(ImageDownloader):
        def process_meta(self, task):
            if task.get("filename"):
                urls[task["filename"]] = task["file_url"]

    with tempfile.TemporaryDirectory() as tmp:
        crawler = BingImageCrawler(downloader_cls=Registro, downloader_threads=hilos, storage={"root_dir": tmp},
                                   log_level=40)
        crawler.crawl(keyword=busqueda, max_num=cuantas, min_size=(LADO_MINIMO, LADO_MINIMO),
                      filters={"type": "photo"})
        for p in sorted(Path(tmp).iterdir()):
            url = urls.get(p.name, p.name)
            c = {"fuente": "bing", "titulo": url, "url_descarga": url, "pagina_fuente": "", "autor": "",
                 "licencia": "desconocida (Bing)"}
            if EXCLUIR.search(urllib.parse.unquote(url)):
                continue
            yield c, None if url in vistos else p.read_bytes()


FUENTES = {"commons": commons, "bing": bing}


# --- Salida ---

def leer_tabla(tabla: Path) -> list[dict]:
    if not tabla.exists():
        return []
    with open(tabla, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def escribir_tabla(tabla: Path, filas: list[dict]):
    with open(tabla, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in COLUMNAS} for r in filas)


def cargar_exclusiones(carpetas: list[str], hashes: Hashes) -> set[str]:
    """Títulos de Commons (de cualquier etiquetas.csv) y hashes de las fotos de `carpetas`."""
    titulos = set()
    for carpeta in map(Path, carpetas):
        for tabla in carpeta.rglob("etiquetas.csv"):
            with open(tabla, encoding="utf-8") as f:
                titulos |= {r["titulo_commons"] for r in csv.DictReader(f) if r.get("titulo_commons")}
        fotos = [p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES]
        for p in fotos:
            im = abrir(p.read_bytes())
            if im is not None:
                hashes.agregar(dhash(im))
        print(f"excluir {carpeta}: {len(fotos)} fotos y {len(titulos)} títulos")
    return titulos


def leer_faltantes(ruta: str) -> dict[str, int]:
    """{alimento: fotos que faltan} de un faltantes.txt (líneas "  alimento: 120", también "(completo)")."""
    metas = {}
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        if m := re.match(r"^\s+([^\s:]+):\s*(\d+)", linea):
            metas[m.group(1)] = int(m.group(2))
    return metas


def siguiente(carpeta: Path, alimento: str) -> int:
    """Siguiente número libre, para no pisar fotos aunque se hayan borrado algunas."""
    numeros = [int(m.group(1)) for p in carpeta.glob(f"web_{alimento}_*.jpg")
               if (m := re.search(r"_(\d+)\.jpg$", p.name))]
    return max(numeros, default=0) + 1


def empaquetar(salida: Path):
    """para_drive/<alimento>/web_<alimento>_1.zip con las fotos que quedan en fotos/<alimento>/."""
    destino = salida / "para_drive"
    shutil.rmtree(destino, ignore_errors=True)
    total = 0
    for alimento in sorted(p.name for p in (salida / "fotos").glob("*") if p.is_dir()):
        fotos = sorted((salida / "fotos" / alimento).glob("web_*.jpg"))
        for k in range(0, len(fotos), FOTOS_POR_ZIP):
            (destino / alimento).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destino / alimento / f"web_{alimento}_{k // FOTOS_POR_ZIP + 1}.zip", "w") as z:
                for p in fotos[k:k + FOTOS_POR_ZIP]:
                    z.write(p, p.name)
        total += len(fotos)
    print(f"\n{total} fotos en {destino}/: sube esas carpetas a la raíz de la carpeta de Drive del equipo.\n"
          "Si una carpeta ya existe en Drive, sube solo el zip dentro de la que ya está.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--salida", default="salida_web")
    ap.add_argument("--plan", default=str(PLAN))
    ap.add_argument("--fuentes", default="bing,commons", help="en orden, separadas por coma: bing, commons")
    ap.add_argument("--excluir", nargs="*", default=[], help="carpetas con fotos que no deben entrar (fotos_prueba)")
    ap.add_argument("--solo", default="", help="alimentos separados por coma (por defecto, todo el plan)")
    ap.add_argument("--meta", type=int, help="fotos por alimento, en lugar de la del plan")
    ap.add_argument("--prioridad", default="", help="prioridades separadas por coma (equipo, sin_fotos, alta, media, secos)")
    ap.add_argument("--faltantes", help="faltantes.txt del notebook: usa sus números como meta de cada alimento")
    ap.add_argument("--pausa", type=float, default=0.5, help="segundos entre descargas de Commons")
    ap.add_argument("--hilos", type=int, default=4, help="descargas a la vez en Bing")
    ap.add_argument("--empaquetar", action="store_true", help="solo rehace para_drive/ con las fotos que quedan")
    a = ap.parse_args()

    with open(a.plan, encoding="utf-8") as f:
        plan = list(csv.DictReader(f))
    if a.solo:
        plan = [f for f in plan if f["alimento"] in a.solo.split(",")]
    if a.prioridad:
        plan = [f for f in plan if f["prioridad"] in a.prioridad.split(",")]
    if a.faltantes:
        metas = leer_faltantes(a.faltantes)
        plan = [{**f, "meta": metas[f["alimento"]]} for f in plan if metas.get(f["alimento"])]
        print(f"faltantes: {len(plan)} alimentos con fotos pendientes, {sum(f['meta'] for f in plan)} fotos")
    salida = Path(a.salida)
    if a.empaquetar:
        return empaquetar(salida)
    fuentes = [x.strip() for x in a.fuentes.split(",") if x.strip()]
    if not fuentes or set(fuentes) - set(FUENTES):
        sys.exit(f"--fuentes debe ser una lista de {', '.join(FUENTES)}")

    tabla = salida / "fuentes_web.csv"
    filas = leer_tabla(tabla)
    for r in filas:  # las que se borraron al revisar quedan marcadas para no volver a bajarlas
        if r.get("estado") != "borrada" and not (salida / r["archivo"]).exists():
            r["estado"] = "borrada"
    hashes = Hashes()
    excluidos = cargar_exclusiones(a.excluir, hashes)
    vistos = excluidos | {r["titulo"] for r in filas}
    for r in filas:
        if r.get("estado") != "borrada" and (im := abrir((salida / r["archivo"]).read_bytes())) is not None:
            hashes.agregar(dhash(im))

    for fila in plan:
        alimento, meta = fila["alimento"], a.meta or int(fila["meta"])
        carpeta = salida / "fotos" / alimento
        carpeta.mkdir(parents=True, exist_ok=True)
        n = sum(1 for r in filas if r["alimento"] == alimento and r.get("estado") != "borrada")
        descartes = {"prueba": 0, "repetida": 0, "estudio": 0, "chica o dañada": 0}
        inicio = time.time()
        for fuente in fuentes:
            for busqueda in fila["busquedas"].split("|"):
                if n >= meta:
                    break
                try:
                    for c, datos in FUENTES[fuente](busqueda, vistos, pausa=a.pausa, hilos=a.hilos,
                                                     cuantas=min(2 * (meta - n) + 10, 300)):
                        if n >= meta:
                            break
                        if datos is None:
                            descartes["prueba" if c["titulo"] in excluidos else "repetida"] += 1
                            continue
                        vistos.add(c["titulo"])
                        im = abrir(datos)
                        if im is None or min(im.size) < LADO_MINIMO:
                            descartes["chica o dañada"] += 1
                            continue
                        if fondo_blanco(im):
                            descartes["estudio"] += 1
                            continue
                        h = dhash(im)
                        if hashes.repetido(h):
                            descartes["repetida"] += 1
                            continue
                        hashes.agregar(h)
                        im.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
                        archivo = carpeta / f"web_{alimento}_{siguiente(carpeta, alimento):04d}.jpg"
                        im.save(archivo, "JPEG", quality=90)
                        filas.append({"archivo": archivo.relative_to(salida).as_posix(), "alimento": alimento,
                                      "prioridad": fila["prioridad"], "busqueda": busqueda, "estado": "", **c})
                        n += 1
                        if n % 20 == 0:
                            escribir_tabla(tabla, filas)
                except Exception as e:
                    print(f"  {fuente} '{busqueda}' interrumpida: {e}")
        escribir_tabla(tabla, filas)
        aviso = "" if n >= meta else "  (no hubo más; completar a mano o con otra fuente)"
        print(f"{alimento:30} {n:4}/{meta} en {time.time() - inicio:4.0f} s  descartes {descartes}{aviso}", flush=True)

    empaquetar(salida)
    print("Antes de subir: revisa salida_web/fotos/ en miniaturas, borra lo que no sea la fruta (o tenga "
          "otras frutas) y corre de nuevo con --empaquetar. Si borraste muchas, corre sin --empaquetar para "
          "completar la meta.")


if __name__ == "__main__":
    main()
