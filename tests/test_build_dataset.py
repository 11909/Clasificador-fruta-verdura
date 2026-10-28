import csv
import shutil

import numpy as np
from PIL import Image

from src.build_dataset import construir, crear_plantilla, preparar_fuentes, resolver_clase
from src.mapping import cargar_mapping

MAPPING = cargar_mapping()


def guardar(ruta, semilla):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)
    Image.fromarray((rng.random((120, 90, 3)) * 255).astype("uint8")).save(ruta)


def test_resolver_clase_limpia_prefijos_y_plurales():
    fuente = {"quitar_prefijos": ["fresh", "rotten"]}
    assert resolver_clase("freshapples", fuente, MAPPING) == ("manzana", "fruta")
    assert resolver_clase("rottentomatoes", fuente, MAPPING) == ("jitomate", "fruta")
    assert resolver_clase("Fresh_Potato", fuente, MAPPING) == ("papa", "verdura")
    assert resolver_clase("Apple Red 2", {}, MAPPING) == ("manzana", "fruta")
    assert resolver_clase("Potato Sweet 1", {}, MAPPING) == ("camote", "verdura")
    assert resolver_clase("Pepino 1", {"renombrar": {"pepino": "pepino dulce"}}, MAPPING) == ("pepino dulce", "fruta")
    assert resolver_clase("Rábano_negro", {}, MAPPING) == ("rabano", "verdura")
    assert resolver_clase("freshxyz", fuente, MAPPING)[1] is None


def test_clases_de_las_fuentes_de_kaggle_quedan_mapeadas():
    # nombres reales de carpeta de freshrotten y PackEat que antes quedaban sin mapeo o en otro alimento
    import yaml
    fuentes = {f["nombre"]: f for f in yaml.safe_load(open("configs/dataset.yaml"))["fuentes"]}
    mapping = cargar_mapping()
    casos = {"freshrotten": {"freshpatato": "papa", "rottentamto": "jitomate"},
             "packeat": {"cabbage_broccoli": "brocoli", "cabbage_cauliflower": "coliflor", "beet_round": "betabel",
                         "raddish_round": "rabano", "salad_green": "lechuga", "tangerine_tangerine": "mandarina",
                         "pepper_chili": "chile", "pepper_sweet_red": "pimiento", "daikon_daikon": "rabano"}}
    for fuente, clases in casos.items():
        for carpeta, alimento in clases.items():
            assert resolver_clase(carpeta, fuentes[fuente], mapping)[0] == alimento, carpeta
    assert fuentes["packeat"]["splits"] == {"train": "train/train", "test": "test/test"}


def test_construir_combina_deduplica_y_reporta(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for i in range(3):
        guardar(a / "Training" / "Apple Red 1" / f"{i}.jpg", i)
        guardar(a / "Training" / "Carrot 1" / f"{i}.jpg", 10 + i)
    guardar(a / "Test" / "Carrot 1" / "t.jpg", 99)
    guardar(a / "Training" / "Mushroom champignon 1" / "m.jpg", 50)
    guardar(b / "Train" / "freshapples" / "x.jpg", 20)
    guardar(b / "Train" / "rottenapples" / "y.jpg", 21)
    guardar(b / "Train" / "freshxyz" / "z.jpg", 22)
    # la misma foto de test aparece en train de otra fuente: debe quedarse solo en test
    (b / "Train" / "freshcarrot").mkdir(parents=True)
    shutil.copy(a / "Test" / "Carrot 1" / "t.jpg", b / "Train" / "freshcarrot" / "copia.jpg")
    # y una foto repetida dentro de train
    shutil.copy(a / "Training" / "Apple Red 1" / "0.jpg", b / "Train" / "freshapples" / "repetida.jpg")

    salida = tmp_path / "out"
    cfg = {"salida": str(salida), "tamano_max": 64, "fuentes": [
        {"nombre": "a", "splits": {"train": str(a / "Training"), "test": str(a / "Test")}},
        {"nombre": "b", "splits": {"train": str(b / "Train")}, "quitar_prefijos": ["fresh", "rotten"]},
    ]}
    reporte = construir(cfg)

    assert reporte["a/test"]["copiadas"] == 1
    assert reporte["a/train"]["copiadas"] == 6 and reporte["a/train"]["excluidas"] == {"hongo": 1}
    assert reporte["b/train"]["copiadas"] == 2 and reporte["b/train"]["duplicadas"] == 2
    assert reporte["b/train"]["sin_mapeo"] == {"freshxyz": 1}

    filas = list(csv.DictReader(open(salida / "manifest.csv", encoding="utf-8")))
    assert len(filas) == 9
    assert {f["clase"] for f in filas if f["fuente"] == "b"} == {"manzana"}
    for f in filas:
        img = Image.open(salida / f["ruta"])
        assert max(img.size) <= 64


def test_fuente_sin_splits_listas_ignorar_y_conteos(tmp_path):
    equipo, grocery, v15 = tmp_path / "equipo", tmp_path / "grocery", tmp_path / "v15"
    for i in range(20):
        guardar(equipo / "oca" / f"{i}.jpg", 100 + i)
    guardar(equipo / "papaya" / "p.jpg", 200)
    guardar(grocery / "train" / "Vegetables" / "Leek" / "l.jpg", 300)
    guardar(grocery / "train" / "Fruit" / "Apple" / "Granny-Smith" / "g.jpg", 301)
    guardar(grocery / "train" / "Packages" / "Juice" / "Bravo-Apple-Juice" / "j.jpg", 302)
    guardar(v15 / "train" / "Brinjal" / "b.jpg", 400)
    guardar(v15 / "validation" / "Radish" / "r.jpg", 401)

    salida = tmp_path / "out"
    cfg = {"salida": str(salida), "tamano_max": 64, "fuentes": [
        {"nombre": "equipo", "ruta": str(equipo), "fraccion_test": 0.3},
        {"nombre": "grocery", "splits": {"train": str(grocery / "train")}, "nivel_clase": 1,
         "ignorar": ["Packages"]},
        {"nombre": "v15", "splits": {"train": [str(v15 / "train"), str(v15 / "validation")]}},
        {"nombre": "falta", "splits": {"train": str(tmp_path / "no-existe")}},
    ]}
    reporte = construir(cfg)

    oca = reporte["equipo/train"]["copiadas"] + reporte["equipo/test"]["copiadas"]
    assert oca == 21 and 0 < reporte["equipo/test"]["copiadas"] < 21
    assert reporte["grocery/train"]["copiadas"] == 2 and reporte["grocery/train"]["sin_mapeo"] == {}
    assert reporte["falta/train"]["copiadas"] == 0

    filas = list(csv.DictReader(open(salida / "manifest.csv", encoding="utf-8")))
    assert {f["clase"] for f in filas if f["fuente"] == "grocery"} == {"poro", "manzana"}
    assert {f["clase"] for f in filas if f["fuente"] == "v15"} == {"berenjena", "rabano"}

    conteos = list(csv.DictReader(open(salida / "conteos.csv", encoding="utf-8")))
    oca_fila = next(f for f in conteos if f["clase"] == "oca")
    assert oca_fila["etiqueta"] == "verdura" and int(oca_fila["train"]) + int(oca_fila["test"]) == 20
    assert [f["etiqueta"] for f in conteos] == sorted(f["etiqueta"] for f in conteos)

    # la asignación a test no cambia al volver a construir
    reporte2 = construir({**cfg, "salida": str(tmp_path / "out2")})
    assert reporte2["equipo/test"]["copiadas"] == reporte["equipo/test"]["copiadas"]


def test_plantilla_crea_una_carpeta_por_alimento_sin_alias(tmp_path):
    nombres = crear_plantilla(str(tmp_path / "fotos"))
    assert (tmp_path / "fotos" / "oca").is_dir() and (tmp_path / "fotos" / "limon_amarillo").is_dir()
    assert "hongo" not in nombres and "berenjena" in nombres and "brinjal" not in nombres
    for nombre in nombres:
        assert MAPPING[nombre] in ("fruta", "verdura")


def test_preparar_fuentes_vuelve_absolutas_las_rutas_y_omite_las_que_fallan(tmp_path):
    raiz = tmp_path / "kaggle" / "dueno" / "fresh"
    (raiz / "dataset" / "dataset" / "Train").mkdir(parents=True)
    (raiz / "Test").mkdir(parents=True)

    def descargar(handle):
        if handle == "dueno/privado":
            raise PermissionError("403")
        return str(raiz)

    cfg = {"fuentes": [
        {"nombre": "fresh", "kaggle": "dueno/fresh", "splits": {"train": "Train", "test": ["Test"]}},
        {"nombre": "equipo", "kaggle": "dueno/privado", "ruta": "."},
        {"nombre": "local", "splits": {"train": "/ya/absoluta"}},
    ]}
    listo = preparar_fuentes(cfg, descargar)

    assert [f["nombre"] for f in listo["fuentes"]] == ["fresh", "local"]
    fresh = listo["fuentes"][0]
    assert fresh["splits"]["train"] == str(raiz / "dataset" / "dataset" / "Train")
    assert fresh["splits"]["test"] == [str(raiz / "Test")]
    assert cfg["fuentes"][0]["splits"]["train"] == "Train"  # no modifica la config original


def test_dividir_equipo_es_estable(tmp_path):
    from src.dataset import dividir_equipo
    for i in range(30):
        guardar(tmp_path / "equipo" / "oca" / f"f{i}.jpg", i)
    train, test = dividir_equipo(str(tmp_path / "equipo"), 0.3, 42)
    assert len(train) + len(test) == 30 and 0 < len(test) < 30
    assert all(m[1] == 1 and m[2] == "oca" for m in train + test)  # verdura
    guardar(tmp_path / "equipo" / "oca" / "nueva.jpg", 99)
    train2, test2 = dividir_equipo(str(tmp_path / "equipo"), 0.3, 42)
    assert {m[0] for m in test} <= {m[0] for m in test2}  # agregar fotos no mueve las anteriores
    assert dividir_equipo(str(tmp_path / "no-existe"), 0.3, 42) == ([], [])
    # zip subido con carpeta envolvente
    guardar(tmp_path / "envuelto" / "fotos" / "nopal" / "a.jpg", 7)
    tr, te = dividir_equipo(str(tmp_path / "envuelto"), 0.3, 42)
    assert [m[2] for m in tr + te] == ["nopal"]


def test_descomprimir_zips_en_raiz_y_en_carpeta_de_alimento(tmp_path):
    import zipfile
    from src.equipo import descomprimir_zips

    drive = tmp_path / "drive"
    (drive / "zanahoria").mkdir(parents=True)
    with zipfile.ZipFile(drive / "ana.zip", "w") as z:  # zip en la raíz con su árbol
        z.writestr("nopal/ana_001.jpg", b"x")
        z.writestr("__MACOSX/nopal/._ana_001.jpg", b"x")
        z.writestr(".DS_Store", b"x")
    with zipfile.ZipFile(drive / "zanahoria" / "luis.zip", "w") as z:  # zip dentro del alimento
        z.writestr("fotos/luis_001.jpg", b"x")
        z.writestr("notas.txt", b"x")
    (raiz_zip,) = descomprimir_zips(str(drive))
    assert sorted(p.relative_to(raiz_zip).as_posix() for p in (tmp_path / "drive_zips" / "ana").rglob("*")) == [
        "nopal", "nopal/ana_001.jpg"]
    assert (drive / "zanahoria" / "luis_001.jpg").exists() and not (drive / "zanahoria" / "notas.txt").exists()


def test_dos_zips_en_una_carpeta_no_se_pisan(tmp_path):
    import zipfile
    from src.equipo import descomprimir_zips

    betabel = tmp_path / "drive" / "betabel"
    betabel.mkdir(parents=True)
    for nombre, contenido in (("web_betabel_1.zip", b"a"), ("web_betabel_2.zip", b"b")):
        with zipfile.ZipFile(betabel / nombre, "w") as z:  # los dos traen 0001.jpg
            z.writestr("0001.jpg", contenido)
            z.writestr("otra/0001.jpg", contenido + b"2")
    for _ in range(2):  # volver a descomprimir no crea copias
        descomprimir_zips(str(tmp_path / "drive"))
    fotos = sorted(p.name for p in betabel.glob("*.jpg"))
    assert len(fotos) == 4 and {(betabel / f).read_bytes() for f in fotos} == {b"a", b"a2", b"b", b"b2"}


def test_fotos_web_repetidas_siguen_siendo_web(tmp_path):
    import zipfile
    from src.dataset import es_web
    from src.equipo import descomprimir_zips

    oca = tmp_path / "drive" / "oca"
    oca.mkdir(parents=True)
    for nombre, contenido in (("oca.zip", b"a"), ("oca_2.zip", b"b")):
        with zipfile.ZipFile(oca / nombre, "w") as z:
            z.writestr("oca/web_oca_0001.jpg", contenido)
    descomprimir_zips(str(tmp_path / "drive"))
    fotos = sorted(oca.glob("*.jpg"))
    assert len(fotos) == 2 and all(es_web(f) for f in fotos)


def test_faltantes_usa_kaggle_y_equipo():
    from collections import Counter
    from src.equipo import faltantes

    plan = [{"persona": "A", "alimento": "oca"}, {"persona": "A", "alimento": "zanahoria"},
            {"persona": "B", "alimento": "rabano"}]
    filas = faltantes(Counter({"zanahoria": 1000, "rabano": 100}), Counter({"oca": 15, "zanahoria": 90}),
                      150, 0.3, 80, plan)
    assert [f["faltan"] for f in filas] == [215 - 15, 0, 80]  # rábano: 50 faltantes / 0.7 = 72 < mínimo 80


def test_tope_distinto_para_verduras():
    from collections import Counter
    from src.dataset import limitar_por_clase
    from src.equipo import faltantes

    muestras = [(f"m{i}", 0, "manzana") for i in range(10)] + [(f"z{i}", 1, "zanahoria") for i in range(10)]
    assert Counter(m[2] for m in limitar_por_clase(muestras, 3, 0, maximo_verdura=6)) == {"manzana": 3, "zanahoria": 6}
    plan = [{"persona": "A", "alimento": "oca"}, {"persona": "A", "alimento": "chile"}]
    assert [f["faltan"] for f in faltantes(Counter(), Counter(), 150, 0.3, 80, plan, 240)] == [343, 215]


def test_muestra_balanceada_80_20():
    from collections import Counter
    from src.dataset import muestra_balanceada

    muestras = ([(f"m{i}", 0, "manzana") for i in range(50)] + [(f"p{i}", 0, "pera") for i in range(50)]
                + [(f"z{i}", 1, "zanahoria") for i in range(30)])
    train, test = muestra_balanceada(muestras, 20, 0.2, 0)
    assert Counter(m[1] for m in train + test) == {0: 20, 1: 20}
    assert Counter(m[1] for m in test) == {0: 4, 1: 4}
    assert not {m[0] for m in train} & {m[0] for m in test}
    assert Counter(m[2] for m in train + test)["manzana"] == 10  # reparte entre clases


def test_muestra_balanceada_toma_primero_drive_y_reparte_por_clase():
    from collections import Counter
    from src.dataset import muestra_balanceada

    kaggle = ([(f"m{i}", 0, "manzana") for i in range(100)] + [(f"z{i}", 1, "zanahoria") for i in range(100)])
    e_train = [(f"e{i}", 1, "oca") for i in range(21)]
    e_test = [(f"t{i}", 1, "oca") for i in range(9)]
    train, test = muestra_balanceada(kaggle, 40, 0.2, 0, (e_train, e_test))
    assert Counter(m[1] for m in train + test) == {0: 40, 1: 40}
    # oca no entra completa: queda igual que zanahoria, con 20, y todas de Drive con su división fija
    assert Counter(m[2] for m in train + test) == {"manzana": 40, "oca": 20, "zanahoria": 20}
    assert {m for m in train if m[2] == "oca"} <= set(e_train) and {m for m in test if m[2] == "oca"} <= set(e_test)
    # Con más cupo entran todas las de Drive y el resto se completa con el dataset combinado
    train, test = muestra_balanceada(kaggle, 100, 0.2, 0, (e_train, e_test))
    assert set(e_train) <= set(train) and set(e_test) <= set(test)
    assert Counter(m[2] for m in train + test) == {"manzana": 100, "oca": 30, "zanahoria": 70}


def test_completar_por_clase_usa_drive_primero_sin_pasar_el_tope():
    from collections import Counter
    from src.dataset import completar_por_clase

    kaggle = ([(f"r{i}", 0, "rambutan") for i in range(100)] + [(f"m{i}", 0, "manzana") for i in range(300)]
              + [(f"z{i}", 1, "zanahoria") for i in range(300)])
    drive = [(f"w{i}", 0, "rambutan") for i in range(70)] + [(f"n{i}", 1, "rabano negro") for i in range(250)]
    train = completar_por_clase(drive, kaggle, 150, 0, maximo_verdura=180)
    por_clase = Counter(m[2] for m in train)
    assert por_clase["rambutan"] == 150 and set(drive[:70]) <= set(train)  # 70 de Drive + 80 del combinado
    assert por_clase["manzana"] == 150 and por_clase["zanahoria"] == 180
    assert por_clase["rabano negro"] == 180  # Drive solo, recortado al tope de verdura


def _drive_falso(tmp_path, monkeypatch, rechazos):
    """gdown simulado: lista kale/a.jpg, kale/b.jpg y oca/oca.zip, y `rechazos[nombre]` dice cuántas
    veces Drive rechaza ese archivo antes de dejarlo bajar."""
    import zipfile
    from types import SimpleNamespace

    import gdown
    from gdown.exceptions import DownloadError
    from PIL import Image

    eq = tmp_path / "eq"
    rutas = ["kale/a.jpg", "kale/b.jpg", "oca/oca.zip"]

    def listar(url, output, skip_download=False, **kwargs):
        assert skip_download
        return [SimpleNamespace(id=r, path=r, local_path=str(eq / r)) for r in rutas]

    def bajar(id, output, **kwargs):
        if rechazos.get(id, 0):
            rechazos[id] -= 1
            raise DownloadError("Too many users have viewed or downloaded this file recently.")
        if id.endswith(".zip"):
            with zipfile.ZipFile(output, "w") as z:  # con carpeta interna, como los del web scraping
                for k in range(3):
                    buf = __import__("io").BytesIO()
                    Image.new("RGB", (20, 20), (k * 80, 0, 0)).save(buf, "JPEG")
                    z.writestr(f"oca/web_oca_000{k}.jpg", buf.getvalue())
        else:
            Image.new("RGB", (20, 20)).save(output)
        return output

    monkeypatch.setattr(gdown, "download_folder", listar)
    monkeypatch.setattr(gdown, "download", bajar)
    return eq


def test_descargar_drive_reintenta_lo_que_drive_rechaza(tmp_path, monkeypatch):
    from collections import Counter
    from src.dataset import dividir_equipo
    from src.equipo import descargar_drive

    # Drive deja de servir después de varias descargas: todo lo que va después falla en la primera vuelta
    eq = _drive_falso(tmp_path, monkeypatch, {"kale/b.jpg": 1, "oca/oca.zip": 2})
    assert descargar_drive("https://drive.google.com/x", str(eq), pausa=0) == [str(eq)]
    tr, te = dividir_equipo(str(eq), 0.3, 42)
    assert Counter(m[2] for m in tr + te) == {"kale": 2, "oca": 3}


def test_descargar_drive_sigue_con_lo_que_bajo(tmp_path, monkeypatch, capsys):
    from src.equipo import descargar_drive

    eq = _drive_falso(tmp_path, monkeypatch, {"oca/oca.zip": 99})
    assert descargar_drive("https://drive.google.com/x", str(eq), rondas=2, pausa=0) == [str(eq)]
    salida = capsys.readouterr().out
    assert "Carpetas afectadas: oca" in salida and "sin fotos: oca" in salida
    assert (eq / "kale" / "b.jpg").exists()


def test_zip_danado_no_tumba_los_demas(tmp_path):
    import zipfile
    from src.equipo import descomprimir_zips

    drive = tmp_path / "drive"
    (drive / "oca").mkdir(parents=True)
    (drive / "nopal").mkdir()
    (drive / "nopal" / "nopal.zip").write_bytes(b"<html>Google Drive - Quota exceeded</html>")
    with zipfile.ZipFile(drive / "oca" / "oca.zip", "w") as z:
        z.writestr("oca/web_oca_0001.jpg", b"x")
    descomprimir_zips(str(drive))
    assert (drive / "oca" / "web_oca_0001.jpg").exists()


def test_descargar_drive_completo_se_detiene_si_faltan(tmp_path, monkeypatch):
    import pytest
    from src.equipo import descargar_drive

    eq = _drive_falso(tmp_path, monkeypatch, {"oca/oca.zip": 99})
    with pytest.raises(RuntimeError, match="oca"):
        descargar_drive("https://drive.google.com/x", str(eq), rondas=2, pausa=0, completo=True)
    # la segunda corrida solo pide lo que falta
    eq = _drive_falso(tmp_path, monkeypatch, {})
    assert descargar_drive("https://drive.google.com/x", str(eq), pausa=0, completo=True) == [str(eq)]


def test_copiar_kaggle_quita_carpeta_envolvente_y_descomprime(tmp_path):
    import zipfile
    from src.equipo import copiar_kaggle

    origen = tmp_path / "input" / "fotos-equipo" / "Fotos equipo"  # como queda el zip que arma Drive
    (origen / "kale").mkdir(parents=True)
    (origen / "oca").mkdir()
    (origen / "kale" / "a.jpg").write_bytes(b"x")
    with zipfile.ZipFile(origen / "oca" / "oca.zip", "w") as z:
        z.writestr("oca/web_oca_0001.jpg", b"y")
    destino = tmp_path / "eq"
    assert copiar_kaggle(str(tmp_path / "input" / "fotos-equipo"), str(destino)) == [str(destino)]
    assert (destino / "kale" / "a.jpg").exists() and (destino / "oca" / "web_oca_0001.jpg").exists()
