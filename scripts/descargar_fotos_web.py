"""Descarga fotos reales de Wikimedia Commons para las frutas que solo tienen Fruits-360.

Uso (desde la raíz del repo, con `pip install -r requirements.txt`):
    python scripts/descargar_fotos_web.py --excluir ruta/a/fotos_prueba
    python scripts/descargar_fotos_web.py --solo rambutan,pitahaya --meta 10   # prueba rápida
Tarda de 2 a 3 horas para las ~6,000 fotos del plan, porque Wikimedia limita la velocidad.

Lee `data/plan_web.csv` (alimento, prioridad, meta de fotos y búsquedas en Commons) y deja
`<salida>/<alimento>/web_<alimento>_001.jpg`, `<salida>/fuentes_web.csv` (título, autor, licencia y página
de cada foto) y `<salida>.zip`. El zip se sube a la raíz de la carpeta de Drive del equipo y el
notebook de entrenamiento lo lee como un árbol de alimentos más (ver docs/detalles.md).

La meta por alimento sale de la misma cuenta que `faltantes.txt`: cada fruta entrena con hasta
MAX_POR_CLASE = 150 fotos, Fruits-360 aporta como máximo 100 tras la limpieza y de las fotos nuevas el
70 % va a train. Así, (150 - 100) / 0.7 = 72, redondeado al mínimo de 80 fotos reales por alimento;
100 en las de prioridad alta, y 215 = 150 / 0.7 en las que no tienen fotos en Kaggle. Ninguna fruta
pasa de las 180 de train de cada verdura, así que las clases quedan parejas.

Se descartan las fotos con fondo blanco de estudio, las de menos de 300 px, los dibujos y los
duplicados (también entre alimentos). `--excluir` recibe carpetas con las fotos de prueba
(fotos_prueba/): ninguna foto que esté ahí, por título de Commons o por hash perceptual, entra al
entrenamiento. Si se vuelve a correr, sigue donde se quedó.
"""
import argparse
import csv
import io
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import EXTENSIONES  # noqa: E402
from src.limpieza import dhash, fondo_blanco  # noqa: E402
from src.mapping import cargar_mapping, etiqueta_de  # noqa: E402

PLAN = Path(__file__).resolve().parent.parent / "data" / "plan_web.csv"
UA = "clasificador-fruta-verdura/1.0 (https://github.com/11909/clasificador-fruta-verdura)"
API = "https://commons.wikimedia.org/w/api.php"
ANCHO = 500        # miniatura estándar de Commons; el dataset guarda las imágenes a 256 px
LADO_MINIMO = 300  # del original
UMBRAL_DUPLICADO = 6
MAX_RESULTADOS = 400  # por búsqueda; más abajo casi todo es ruido
# Títulos que casi nunca son la fruta sola: dibujos, láminas, plantas en flor, platillos y bebidas
EXCLUIR = re.compile(
    r"(drawing|illustration|diagram|logo|\bmap\b|icon|painting|stamp|herbari|fruits[-_ ]?360|engraving|"
    r"lithograph|\bplate\b|tafel|k[oö]hler|botanical|\bflowers?\b|blossom|\bflor(es)?\b|\btrees?\b|"
    r"\bleaf\b|leaves|seedling|\bplants?\b|\bjuice|\bjam\b|\bcake|\bpie\b|salad|smoothie|dessert|"
    r"ice[ _-]?cream|cocktail|\bwine\b|liqueur|recipe|\bdish\b|\bcurry\b|stall[ _-]?sign|\bcoins?\b|"
    r"\bbooks?\b|poster|\blabel|\bpdf\b|\btiff?\b|\bsvg\b)", re.I)


def pedir(url: str, intentos: int = 5) -> bytes:
    """GET con User-Agent propio y espera creciente si Wikimedia pide bajar la velocidad (429/503)."""
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


def candidatos(busqueda: str):
    """Archivos de Commons para `busqueda`, en el orden de relevancia, de 50 en 50."""
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
            yield {"titulo_commons": p["title"], "url_descarga": info.get("thumburl") or info["url"],
                   "pagina_fuente": info.get("descriptionurl", ""), "autor": autor[:200],
                   "licencia": meta.get("LicenseShortName", {}).get("value", "")}
        if "continue" not in datos:
            return
        time.sleep(1)


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


def cargar_exclusiones(carpetas: list[str], hashes: Hashes) -> set[str]:
    """Títulos de Commons (de cualquier etiquetas.csv o fuentes*.csv) y hashes de las fotos de `carpetas`."""
    titulos = set()
    for carpeta in map(Path, carpetas):
        for tabla in list(carpeta.rglob("etiquetas.csv")) + list(carpeta.rglob("fuentes*.csv")):
            with open(tabla, encoding="utf-8") as f:
                titulos |= {r["titulo_commons"] for r in csv.DictReader(f) if r.get("titulo_commons")}
        fotos = [p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES]
        for p in fotos:
            hashes.agregar(dhash(str(p)))
        print(f"excluir {carpeta}: {len(fotos)} fotos")
    return titulos


def guardar(datos: bytes, destino: Path) -> bool:
    """Guarda la imagen como JPEG si abre bien y no es de estudio; False si se descarta."""
    try:
        with Image.open(io.BytesIO(datos)) as im:
            im = im.convert("RGB")
    except Exception:
        return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    im.save(destino, "JPEG", quality=90)
    if fondo_blanco(str(destino)):
        destino.unlink()
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--salida", default="fotos_web")
    ap.add_argument("--excluir", nargs="*", default=[], help="carpetas con fotos que no deben entrar (fotos_prueba)")
    ap.add_argument("--solo", default="", help="alimentos separados por coma (por defecto, todo el plan)")
    ap.add_argument("--meta", type=int, help="fotos por alimento, en lugar de la del plan")
    ap.add_argument("--pausa", type=float, default=1.0, help="segundos entre descargas")
    ap.add_argument("--sin-zip", action="store_true")
    a = ap.parse_args()

    with open(PLAN, encoding="utf-8") as f:
        plan = list(csv.DictReader(f))
    mapping = cargar_mapping()
    for fila in plan:
        if etiqueta_de(fila["alimento"], mapping) is None:
            sys.exit(f"{fila['alimento']} no está en data/mapping.csv")
    if a.solo:
        plan = [f for f in plan if f["alimento"] in a.solo.split(",")]

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    tabla = salida / "fuentes_web.csv"
    columnas = ["archivo", "alimento", "prioridad", "busqueda", "titulo_commons", "autor", "licencia",
                "pagina_fuente", "url_descarga"]
    filas = []
    if tabla.exists():
        with open(tabla, encoding="utf-8") as f:
            filas = [r for r in csv.DictReader(f) if (salida / r["archivo"]).exists()]
    hashes = Hashes()
    excluidos = cargar_exclusiones(a.excluir, hashes)
    vistos = excluidos | {r["titulo_commons"] for r in filas}
    for r in filas:
        hashes.agregar(dhash(str(salida / r["archivo"])))

    def escribir_tabla():
        with open(tabla, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columnas)
            w.writeheader()
            w.writerows(filas)

    for fila in plan:
        alimento, meta = fila["alimento"], a.meta or int(fila["meta"])
        n = sum(1 for r in filas if r["alimento"] == alimento)
        descartes = {"prueba": 0, "duplicado": 0, "estudio": 0}
        for busqueda in fila["busquedas"].split("|"):
            if n >= meta:
                break
            try:
                for c in candidatos(busqueda):
                    if n >= meta:
                        break
                    if c["titulo_commons"] in vistos:
                        descartes["prueba" if c["titulo_commons"] in excluidos else "duplicado"] += 1
                        continue
                    vistos.add(c["titulo_commons"])
                    try:
                        datos = pedir(c["url_descarga"])
                    except Exception as e:
                        print(f"  falló {c['url_descarga']}: {e}")
                        continue
                    finally:
                        time.sleep(a.pausa)
                    archivo = Path(alimento) / f"web_{alimento}_{n + 1:03d}.jpg"
                    if not guardar(datos, salida / archivo):
                        descartes["estudio"] += 1
                        continue
                    h = dhash(str(salida / archivo))
                    if hashes.repetido(h):
                        (salida / archivo).unlink()
                        descartes["duplicado"] += 1
                        continue
                    hashes.agregar(h)
                    filas.append({"archivo": archivo.as_posix(), "alimento": alimento, "prioridad": fila["prioridad"],
                                  "busqueda": busqueda, **c})
                    n += 1
                    if n % 20 == 0:
                        escribir_tabla()
            except Exception as e:
                print(f"  búsqueda '{busqueda}' interrumpida: {e}")
        escribir_tabla()
        aviso = "" if n >= meta else "  (Commons no tiene más; completar a mano)"
        print(f"{alimento:30} {n:4}/{meta}  descartes {descartes}{aviso}", flush=True)

    total = sum(1 for r in filas if any(r["alimento"] == f["alimento"] for f in plan))
    print(f"\n{total} fotos en {salida}/")
    if not a.sin_zip:
        shutil.make_archive(str(salida), "zip", salida.parent, salida.name)
        print(f"zip: {salida}.zip (súbelo a la raíz de la carpeta de Drive del equipo)")


if __name__ == "__main__":
    main()
