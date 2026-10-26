import csv
import importlib.util
import io
import json
import shutil
from pathlib import Path

from PIL import Image

from src.equipo import PLAN_WEB_PATH, escribir_faltantes, faltantes, leer_plan_web
from src.mapping import cargar_mapping, etiqueta_de

RAIZ = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("web_scrapping", RAIZ / "scripts" / "web_scrapping.py")
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


def test_plan_web_son_frutas_del_mapping_con_meta_pareja():
    mapping = cargar_mapping()
    plan = list(csv.DictReader(open(PLAN_WEB_PATH, encoding="utf-8")))
    assert len({f["alimento"] for f in plan}) == len(plan)
    from src.equipo import leer_plan
    del_equipo = {p["alimento"] for p in leer_plan()}
    assert {f["alimento"] for f in plan if f["prioridad"] == "equipo"} <= del_equipo
    assert all(len(f["busquedas"].split("|")) >= 2 for f in plan)
    for f in [f for f in plan if f["prioridad"] != "equipo"]:
        assert etiqueta_de(f["alimento"], mapping) == "fruta", f["alimento"]
        # con 100 de Fruits-360 y 70 % a train, ninguna fruta pasa de las 180 de train de una verdura
        assert 80 <= int(f["meta"]) and 100 * (f["prioridad"] != "sin_fotos") + 0.7 * int(f["meta"]) <= 180
        assert len(f["busquedas"].split("|")) >= 2


def test_faltantes_lista_las_frutas_web_aparte(tmp_path):
    from collections import Counter
    assert not any(f["prioridad"] == "equipo" for f in leer_plan_web())  # esas ya salen en su persona
    plan = [{"persona": "A", "alimento": "oca"}] + [f for f in leer_plan_web() if f["alimento"] == "rambutan"]
    filas = faltantes(Counter({"rambutan": 490}), Counter({"rambutan": 30}), 150, 0.3, 80, plan)
    assert [f["faltan"] for f in filas] == [215, 50]
    texto = escribir_faltantes(filas, tmp_path / "faltantes.txt")
    assert texto.index("Persona A") < texto.index("Fotos de internet") and "rambutan: 50" in texto


def jpeg(semilla, tam=(600, 400), borde_blanco=False):
    """Foto distinta por semilla (ruido grueso, para que su hash perceptual no se parezca al de otras)."""
    import numpy as np
    ruido = np.random.default_rng(semilla).integers(0, 200, (8, 12, 3), dtype=np.uint8)
    im = Image.fromarray(ruido).resize(tam, Image.NEAREST)
    if borde_blanco:
        fondo = Image.new("RGB", tam, (255, 255, 255))
        fondo.paste(im.resize((300, 150)), (150, 125))
        im = fondo
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return buf.getvalue()


def test_candidatos_filtra_titulos_y_originales_chicos(monkeypatch):
    paginas = {str(i): {"title": t, "index": i, "imageinfo": [{"mime": "image/jpeg", "width": w, "height": 600,
                        "url": f"https://x/{i}.jpg", "thumburl": f"https://x/t{i}.jpg",
                        "descriptionurl": "", "extmetadata": {}}]}
               for i, (t, w) in enumerate([("File:Rambutan market.jpg", 1200), ("File:Rambutan tree.jpg", 1200),
                                          ("File:Rambutan small.jpg", 400), ("File:Rambutan drawing.jpg", 900)])}
    monkeypatch.setattr(web, "pedir", lambda url: json.dumps({"query": {"pages": paginas}}).encode())
    assert [c["titulo"] for c in web.candidatos_commons("rambutan")] == ["File:Rambutan market.jpg"]


def test_descarga_completa_lista_para_drive(monkeypatch, tmp_path):
    prueba = tmp_path / "fotos_prueba"
    (prueba / "rambutan").mkdir(parents=True)
    (prueba / "rambutan" / "r.jpg").write_bytes(jpeg(1))
    with open(prueba / "etiquetas.csv", "w", encoding="utf-8") as f:
        f.write("archivo,titulo_commons\nrambutan/x.jpg,File:Usada en prueba.jpg\n")
    fotos = {"File:Usada en prueba.jpg": jpeg(2), "File:Igual a prueba.jpg": jpeg(1),
             "File:Estudio.jpg": jpeg(3, borde_blanco=True), "File:Buena.jpg": jpeg(4),
             "File:Otra buena.jpg": jpeg(5), "File:Tercera.jpg": jpeg(6)}

    def falsa(busqueda, vistos, **_):
        for t, datos in fotos.items():
            c = {"fuente": "commons", "titulo": t, "url_descarga": t, "pagina_fuente": "", "autor": "",
                 "licencia": "CC BY"}
            yield c, None if t in vistos else datos

    monkeypatch.setitem(web.FUENTES, "commons", falsa)
    salida = tmp_path / "salida_web"
    args = ["x", "--salida", str(salida), "--fuentes", "commons", "--solo", "rambutan", "--meta", "2",
            "--excluir", str(prueba)]
    monkeypatch.setattr("sys.argv", args)
    web.main()
    filas = list(csv.DictReader(open(salida / "fuentes_web.csv", encoding="utf-8")))
    assert [f["titulo"] for f in filas] == ["File:Buena.jpg", "File:Otra buena.jpg"]
    fotos_dir = salida / "fotos" / "rambutan"
    assert sorted(p.name for p in fotos_dir.iterdir()) == ["web_rambutan_0001.jpg", "web_rambutan_0002.jpg"]

    # Al revisar se borra una; al volver a correr no se baja otra vez y se completa con la siguiente
    (fotos_dir / "web_rambutan_0001.jpg").unlink()
    web.main()
    filas = list(csv.DictReader(open(salida / "fuentes_web.csv", encoding="utf-8")))
    assert [(f["titulo"], f["estado"]) for f in filas] == [
        ("File:Buena.jpg", "borrada"), ("File:Otra buena.jpg", ""), ("File:Tercera.jpg", "")]
    assert sorted(p.name for p in fotos_dir.iterdir()) == ["web_rambutan_0002.jpg", "web_rambutan_0003.jpg"]

    # para_drive/<fruta>/<zip> se sube tal cual a Drive, y el notebook lo lee como fotos de esa fruta
    zips = sorted(p.relative_to(salida / "para_drive").as_posix() for p in (salida / "para_drive").rglob("*.zip"))
    assert zips == ["rambutan/web_rambutan_1.zip"]
    from src.dataset import dividir_equipo, es_web
    from src.equipo import descomprimir_zips
    drive = tmp_path / "drive"
    shutil.copytree(salida / "para_drive", drive)
    descomprimir_zips(str(drive))
    a, b = dividir_equipo(str(drive), 0.3, 42)
    assert {m[2] for m in a + b} == {"rambutan"} and len(a + b) == 2 and all(es_web(m[0]) for m in a + b)


def test_fotos_web_false_las_deja_fuera(tmp_path):
    from src.train import equipo
    for nombre in ("web_rambutan_0001.jpg", "ana_001.jpg"):
        (tmp_path / "rambutan").mkdir(exist_ok=True)
        Image.new("RGB", (30, 30)).save(tmp_path / "rambutan" / nombre)
    d = {"equipo_dir": [str(tmp_path)], "fraccion_test_equipo": 0.0}
    assert len(equipo(d, [], 0)[0]) == 2
    assert [Path(m[0]).name for m in equipo({**d, "fotos_web": False}, [], 0)[0]] == ["ana_001.jpg"]


def test_leer_faltantes_del_notebook_y_de_drive(tmp_path):
    texto = ("Persona A (faltan 300):\n  oca: 200\n  zanahoria: 0  (completo)\n\n"
             "Fotos de internet (faltan 80):\n  Prioridad media (80 c/u):\n    salak: 80\n")
    (tmp_path / "faltantes.txt").write_text(texto, encoding="utf-8")
    assert web.leer_faltantes(str(tmp_path / "faltantes.txt")) == {"oca": 200, "zanahoria": 0, "salak": 80}
