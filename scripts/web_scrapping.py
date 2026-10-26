"""Descarga fotos reales de internet para las frutas que solo tienen Fruits-360 y las deja listas para Drive.

Uso (Python 3.10+, `pip install pillow numpy icrawler torch transformers`):
    python web_scrapping.py --solo rambutan --meta 10           # prueba rápida
    python web_scrapping.py --excluir ruta/a/fotos_prueba       # todo el plan
    python web_scrapping.py --faltantes faltantes.txt           # metas con los conteos reales de Drive
    python web_scrapping.py --revisar                           # limpia con el filtro las fotos ya bajadas
    python web_scrapping.py --empaquetar                        # opcional: zips por carpeta en para_drive/

Lee `plan_web.csv` (junto a este archivo, o `data/plan_web.csv` del repo): alimento, prioridad, meta de
fotos y búsquedas. Trae dos grupos:
    frutas con solo Fruits-360 (prioridad sin_fotos, alta, media, secos): ~6,000 fotos
    alimentos del plan del equipo que aún faltan en Drive (prioridad equipo): ~12,900 fotos, según
    faltantes.txt menos el lote de las Personas A, B y C. Con `--faltantes faltantes.txt` (el que escribe
    el notebook en Output, con los conteos reales de Drive) la meta de cada alimento sale de ahí.
    `--prioridad equipo` baja solo ese grupo; `--prioridad sin_fotos,alta,media,secos`, solo las frutas.
Deja:
    salida_web/fotos/<alimento>/web_<alimento>_0001.jpg   se revisan en miniaturas y se suben tal cual a Drive
    salida_web/fuentes_web.csv                            origen, autor y licencia de cada foto
Con `--empaquetar` además arma salida_web/para_drive/<alimento>/web_<alimento>_1.zip, por si Drive tarda
en subir o listar carpetas con muchas fotos sueltas.

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
duplicados (también entre alimentos). Las búsquedas de Bing llevan palabras a excluir (`-ave`, `-receta`
y las de la columna `excluir` del plan, como `-ganso` en oca). Además, si están instalados torch y
transformers, un modelo CLIP revisa cada foto y descarta las que no son del alimento: aves, personas,
platillos, dibujos, edificios... (`--sin-filtro` lo apaga). Para limpiar fotos ya bajadas:
`--revisar` pasa el filtro por salida_web/fotos/ y mueve lo que no pase a salida_web/descartadas/. `--excluir` recibe la carpeta de fotos de prueba (fotos_prueba/):
ninguna foto que esté ahí, por título de Commons o por hash perceptual, entra al entrenamiento.
Si se corta o se vuelve a correr, sigue donde se quedó; las fotos que borres al revisar no se vuelven
a bajar.
"""
import argparse
import csv
import io
import json
import logging
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
# Se agregan con "-" a las búsquedas de Bing, además de la columna `excluir` del plan
EXCLUIR_BING = ["ave", "bird", "dibujo", "drawing", "clipart", "cartoon", "receta", "recipe", "logo", "ilustracion"]
# Filtro CLIP: la foto pasa si "el alimento" o "otra fruta o verdura" gana a todo lo que no es comida cruda.
# La segunda opción absorbe los alimentos raros que CLIP no conoce bien (oca, chaya...)
CLIP = "openai/clip-vit-base-patch32"
UMBRAL_CLIP = 0.6  # calibrado con fotos reales: oca 1.00, ganso 0.00, kiwi ave 0.22, romeritos en mole 0.59
NO_ALIMENTO = ["a bird", "an animal", "a person", "a cooked dish or meal", "a plate or bowl of prepared food",
               "a soup, stew or food in sauce", "a salad or chopped food", "a drink", "a dessert or baked food",
               "a drawing, cartoon or illustration", "a logo, text or a poster", "a packaged product in a box or bag",
               "a building, street or landscape", "a flowering plant or a tree without fruit",
               "a cactus or a whole plant growing in a garden or field", "a fish or seafood",
               "a toy, a phone or another object"]
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


def bing(busqueda: str, vistos: set, cuantas: int, hilos: int = 4, excluir: str = "", **_):
    """Bing Imágenes con icrawler: baja `cuantas` fotos a la vez en una carpeta temporal. La búsqueda
    lleva las palabras de EXCLUIR_BING y de `excluir` con "-" para que Bing las deje fuera."""
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
        # CRITICAL: icrawler avisa cada página que no responde ("Exception caught when downloading file");
        # esas fotos solo se saltan, no hace falta verlo
        crawler = BingImageCrawler(downloader_cls=Registro, downloader_threads=hilos, storage={"root_dir": tmp},
                                   log_level=logging.CRITICAL)
        negativas = " ".join(f"-{w}" for w in EXCLUIR_BING + excluir.split())
        crawler.crawl(keyword=f"{busqueda} {negativas}", max_num=cuantas, min_size=(LADO_MINIMO, LADO_MINIMO),
                      filters={"type": "photo"})
        for p in sorted(Path(tmp).iterdir()):
            url = urls.get(p.name, p.name)
            c = {"fuente": "bing", "titulo": url, "url_descarga": url, "pagina_fuente": "", "autor": "",
                 "licencia": "desconocida (Bing)"}
            if EXCLUIR.search(urllib.parse.unquote(url)):
                continue
            yield c, None if url in vistos else p.read_bytes()


FUENTES = {"commons": commons, "bing": bing}


class Filtro:
    """CLIP sin entrenar: compara la foto con "una foto de <alimento>" y con cosas que no son el alimento."""

    def __init__(self, umbral: float = UMBRAL_CLIP, nucleos: int = 2):
        import torch
        torch.set_num_threads(nucleos)  # pocos núcleos: más lento, pero la compu no se calienta tanto
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.modelo = CLIPModel.from_pretrained(CLIP).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP)
        self.textos = {}
        self.umbral = umbral

    def _texto(self, nombre_en: str):
        if nombre_en not in self.textos:
            frases = [f"a photo of {nombre_en}", "a photo of a different fresh fruit, vegetable or nut"]
            frases += [f"a photo of {x}" for x in NO_ALIMENTO]
            self.textos[nombre_en] = self.proc(text=frases, return_tensors="pt", padding=True)
        return self.textos[nombre_en]

    def probabilidad(self, im: Image.Image, nombre_en: str) -> float:
        """Probabilidad de que la foto sea el alimento (o al menos un alimento crudo)."""
        with self.torch.no_grad():
            salida = self.modelo(**self._texto(nombre_en), pixel_values=self.proc(images=im, return_tensors="pt")["pixel_values"])
        p = salida.logits_per_image.softmax(dim=-1)[0]
        return float(p[0] + p[1])

    def pasa(self, im: Image.Image, nombre_en: str) -> bool:
        return self.probabilidad(im, nombre_en) >= self.umbral


def crear_filtro(apagado: bool, umbral: float = UMBRAL_CLIP, nucleos: int = 2) -> "Filtro | None":
    if apagado:
        return None
    try:
        filtro = Filtro(umbral, nucleos)
        print("Filtro CLIP activo: se descartan las fotos que no son del alimento.")
        return filtro
    except ImportError:
        print("Aviso: sin filtro CLIP (pip install torch transformers para activarlo). Revisa bien las fotos.")
        return None


def revisar(salida: Path, plan: list[dict], filtro: Filtro, filas: list[dict]):
    """Pasa el filtro por las fotos ya bajadas y mueve a descartadas/ las que no son del alimento."""
    nombres = {f["alimento"]: f["nombre_en"] for f in plan}
    movidas = 0
    for r in filas:
        ruta = salida / r["archivo"]
        if r.get("estado") or r["alimento"] not in nombres or not ruta.exists():
            continue
        im = abrir(ruta.read_bytes())
        if im is None or not filtro.pasa(im, nombres[r["alimento"]]):
            destino = salida / "descartadas" / r["archivo"].split("/", 1)[1]
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ruta), destino)
            r["estado"] = "descartada"
            movidas += 1
    print(f"revisión: {movidas} fotos movidas a {salida / 'descartadas'}/")


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
    ap.add_argument("--nucleos", type=int, default=2,
                    help="núcleos del procesador para el filtro (menos = más lento pero calienta menos)")
    ap.add_argument("--empaquetar", action="store_true", help="solo rehace para_drive/ con las fotos que quedan")
    ap.add_argument("--revisar", action="store_true", help="pasa el filtro CLIP por las fotos ya bajadas")
    ap.add_argument("--sin-filtro", action="store_true", help="no usar el filtro CLIP")
    ap.add_argument("--umbral", type=float, default=UMBRAL_CLIP,
                    help="qué tan seguro debe estar el filtro (más alto = descarta más)")
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
        if not r.get("estado") and not (salida / r["archivo"]).exists():
            r["estado"] = "borrada"
    filtro = crear_filtro(a.sin_filtro, a.umbral, a.nucleos)
    if a.revisar:
        if filtro is None:
            sys.exit("--revisar necesita el filtro CLIP: pip install torch transformers")
        revisar(salida, plan, filtro, filas)
        return escribir_tabla(tabla, filas)
    hashes = Hashes()
    excluidos = cargar_exclusiones(a.excluir, hashes)
    vistos = excluidos | {r["titulo"] for r in filas}
    for r in filas:
        if not r.get("estado") and (im := abrir((salida / r["archivo"]).read_bytes())) is not None:
            hashes.agregar(dhash(im))

    for fila in plan:
        alimento, meta = fila["alimento"], a.meta or int(fila["meta"])
        carpeta = salida / "fotos" / alimento
        carpeta.mkdir(parents=True, exist_ok=True)
        n = sum(1 for r in filas if r["alimento"] == alimento and not r.get("estado"))
        descartes = {"prueba": 0, "repetida": 0, "estudio": 0, "chica o dañada": 0, "no es el alimento": 0}
        no_es = re.compile("|".join(rf"\b{re.escape(w)}" for w in fila.get("excluir", "").split()) or "$^", re.I)
        inicio = time.time()
        for fuente in fuentes:
            for busqueda in fila["busquedas"].split("|"):
                if n >= meta:
                    break
                try:
                    for c, datos in FUENTES[fuente](busqueda, vistos, pausa=a.pausa, hilos=a.hilos,
                                                     excluir=fila.get("excluir", ""),
                                                     cuantas=min(2 * (meta - n) + 10, 300)):
                        if n >= meta:
                            break
                        if datos is None:
                            descartes["prueba" if c["titulo"] in excluidos else "repetida"] += 1
                            continue
                        vistos.add(c["titulo"])
                        if no_es.search(urllib.parse.unquote(c["titulo"])):
                            descartes["no es el alimento"] += 1
                            continue
                        im = abrir(datos)
                        if im is None or min(im.size) < LADO_MINIMO:
                            descartes["chica o dañada"] += 1
                            continue
                        if fondo_blanco(im):
                            descartes["estudio"] += 1
                            continue
                        if filtro and not filtro.pasa(im, fila.get("nombre_en") or alimento.replace("_", " ")):
                            descartes["no es el alimento"] += 1
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

    total = sum(1 for r in filas if not r.get("estado"))
    print(f"\n{total} fotos en {salida / 'fotos'}/. Revisa en miniaturas, borra lo que no sea el alimento "
          "(o tenga otros alimentos) y sube esas carpetas a la raíz de la carpeta de Drive del equipo. Si "
          "borraste muchas, corre otra vez para completar la meta.")


if __name__ == "__main__":
    main()
