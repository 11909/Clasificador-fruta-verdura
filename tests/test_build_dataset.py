import csv
import shutil

import numpy as np
from PIL import Image

from src.build_dataset import construir, crear_plantilla, resolver_clase
from src.mapping import cargar_mapping

MAPPING = cargar_mapping()


def guardar(ruta, semilla):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)
    Image.fromarray((rng.random((120, 90, 3)) * 255).astype("uint8")).save(ruta)


def test_resolver_clase_limpia_prefijos_y_plurales():
    fuente = {"quitar_prefijos": ["fresh", "rotten"], "renombrar": {"bitter gourd": "bitter gourd"}}
    assert resolver_clase("freshapples", fuente, MAPPING) == ("apple", "fruta")
    assert resolver_clase("rottentomatoes", fuente, MAPPING) == ("tomato", "fruta")
    assert resolver_clase("Fresh_Potato", fuente, MAPPING) == ("potato", "verdura")
    assert resolver_clase("Apple Red 2", {}, MAPPING) == ("apple red", "fruta")
    assert resolver_clase("freshxyz", fuente, MAPPING)[1] is None


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
    assert reporte["a/train"]["copiadas"] == 6 and reporte["a/train"]["excluidas"] == {"mushroom champignon": 1}
    assert reporte["b/train"]["copiadas"] == 2 and reporte["b/train"]["duplicadas"] == 2
    assert reporte["b/train"]["sin_mapeo"] == {"freshxyz": 1}

    filas = list(csv.DictReader(open(salida / "manifest.csv", encoding="utf-8")))
    assert len(filas) == 9
    assert {f["clase"] for f in filas if f["fuente"] == "b"} == {"apple"}
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
        {"nombre": "v15", "splits": {"train": [str(v15 / "train"), str(v15 / "validation")]},
         "renombrar": {"brinjal": "eggplant"}},
        {"nombre": "falta", "splits": {"train": str(tmp_path / "no-existe")}},
    ]}
    reporte = construir(cfg)

    oca = reporte["equipo/train"]["copiadas"] + reporte["equipo/test"]["copiadas"]
    assert oca == 21 and 0 < reporte["equipo/test"]["copiadas"] < 21
    assert reporte["grocery/train"]["copiadas"] == 2 and reporte["grocery/train"]["sin_mapeo"] == {}
    assert reporte["falta/train"]["copiadas"] == 0

    filas = list(csv.DictReader(open(salida / "manifest.csv", encoding="utf-8")))
    assert {f["clase"] for f in filas if f["fuente"] == "grocery"} == {"leek", "apple"}
    assert {f["clase"] for f in filas if f["fuente"] == "v15"} == {"eggplant", "radish"}

    conteos = list(csv.DictReader(open(salida / "conteos.csv", encoding="utf-8")))
    oca_fila = next(f for f in conteos if f["clase"] == "oca")
    assert oca_fila["etiqueta"] == "verdura" and int(oca_fila["train"]) + int(oca_fila["test"]) == 20
    assert [f["etiqueta"] for f in conteos] == sorted(f["etiqueta"] for f in conteos)

    # la asignación a test no cambia al volver a construir
    reporte2 = construir({**cfg, "salida": str(tmp_path / "out2")})
    assert reporte2["equipo/test"]["copiadas"] == reporte["equipo/test"]["copiadas"]


def test_plantilla_crea_una_carpeta_por_alimento_sin_alias(tmp_path):
    nombres = crear_plantilla(str(tmp_path / "fotos"))
    assert (tmp_path / "fotos" / "oca").is_dir() and (tmp_path / "fotos" / "sweet_potato").is_dir()
    assert "brinjal" not in nombres and "mushroom" not in nombres and "eggplant" in nombres
    for nombre in nombres:
        assert MAPPING[nombre] in ("fruta", "verdura")
