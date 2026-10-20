import csv
import shutil

import numpy as np
from PIL import Image

from src.build_dataset import construir, resolver_clase
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
