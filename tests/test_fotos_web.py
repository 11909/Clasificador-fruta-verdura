import csv
import importlib.util
import io
import json
from pathlib import Path

from PIL import Image

from src.equipo import PLAN_WEB_PATH, escribir_faltantes, faltantes, leer_plan_web
from src.mapping import cargar_mapping, etiqueta_de

RAIZ = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("descargar_fotos_web", RAIZ / "scripts" / "descargar_fotos_web.py")
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


def test_plan_web_son_frutas_del_mapping_con_meta_pareja():
    mapping = cargar_mapping()
    plan = list(csv.DictReader(open(PLAN_WEB_PATH, encoding="utf-8")))
    assert len({f["alimento"] for f in plan}) == len(plan)
    for f in plan:
        assert etiqueta_de(f["alimento"], mapping) == "fruta", f["alimento"]
        # con 100 de Fruits-360 y 70 % a train, ninguna fruta pasa de las 180 de train de una verdura
        assert 80 <= int(f["meta"]) and 100 * (f["prioridad"] != "sin_fotos") + 0.7 * int(f["meta"]) <= 180
        assert len(f["busquedas"].split("|")) >= 2


def test_faltantes_lista_las_frutas_web_aparte(tmp_path):
    from collections import Counter
    plan = [{"persona": "A", "alimento": "oca"}] + [f for f in leer_plan_web() if f["alimento"] == "rambutan"]
    filas = faltantes(Counter({"rambutan": 490}), Counter({"rambutan": 30}), 150, 0.3, 80, plan)
    assert [f["faltan"] for f in filas] == [215, 50]
    texto = escribir_faltantes(filas, tmp_path / "faltantes.txt")
    assert texto.index("Persona A") < texto.index("Fotos de internet") and "rambutan: 50" in texto


def jpeg(color, tam=(600, 400), borde_blanco=False):
    im = Image.new("RGB", tam, color)
    if borde_blanco:
        im = Image.new("RGB", tam, (255, 255, 255))
        im.paste(color, (100, 100, 500, 300))
    else:
        for x in range(0, tam[0], 40):  # textura para que el hash no sea trivial
            im.paste((color[0] // 2, color[1], color[2]), (x, 0, x + 20, tam[1]))
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
    assert [c["titulo_commons"] for c in web.candidatos("rambutan")] == ["File:Rambutan market.jpg"]


def test_descarga_completa_sin_fotos_de_prueba_ni_duplicados(monkeypatch, tmp_path):
    prueba = tmp_path / "fotos_prueba"
    (prueba / "rambutan").mkdir(parents=True)
    (prueba / "rambutan" / "r.jpg").write_bytes(jpeg((200, 30, 30)))
    with open(prueba / "etiquetas.csv", "w", encoding="utf-8") as f:
        f.write("archivo,titulo_commons\nrambutan/x.jpg,File:Usada en prueba.jpg\n")
    fotos = {"File:Usada en prueba.jpg": jpeg((10, 10, 200)), "File:Igual a prueba.jpg": jpeg((200, 30, 30)),
             "File:Estudio.jpg": jpeg((200, 30, 30), borde_blanco=True), "File:Buena.jpg": jpeg((30, 160, 40)),
             "File:Otra buena.jpg": jpeg((240, 200, 20))}
    monkeypatch.setattr(web, "candidatos", lambda b: iter(
        [{"titulo_commons": t, "url_descarga": t, "pagina_fuente": "", "autor": "", "licencia": "CC BY"} for t in fotos]))
    monkeypatch.setattr(web, "pedir", lambda url: fotos[url])
    salida = tmp_path / "fotos_web"
    monkeypatch.setattr("sys.argv", ["x", "--salida", str(salida), "--solo", "rambutan", "--meta", "5",
                                     "--pausa", "0", "--excluir", str(prueba)])
    web.main()
    filas = list(csv.DictReader(open(salida / "fuentes_web.csv", encoding="utf-8")))
    assert [f["titulo_commons"] for f in filas] == ["File:Buena.jpg", "File:Otra buena.jpg"]
    assert sorted(p.name for p in (salida / "rambutan").iterdir()) == ["web_rambutan_001.jpg", "web_rambutan_002.jpg"]
    assert (tmp_path / "fotos_web.zip").exists()

    # El zip en la raíz de Drive se lee como un árbol de alimentos más
    from src.dataset import dividir_equipo
    from src.equipo import descomprimir_zips
    drive = tmp_path / "drive"
    drive.mkdir()
    (tmp_path / "fotos_web.zip").rename(drive / "fotos_web.zip")
    raices = descomprimir_zips(str(drive))
    a, b = dividir_equipo(raices[0], 0.3, 42)
    assert Path(raices[0]).name == "fotos_web" and {m[2] for m in a + b} == {"rambutan"} and len(a + b) == 2
