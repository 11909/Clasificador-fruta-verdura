import csv

import pytest
from PIL import Image

from src.dataset import fuente_de


def test_fuente_de():
    assert fuente_de("/x/train/mango/fruits360_0123456789ab.jpg") == "fruits360"
    assert fuente_de("/x/train/col/vegetales15_abcdef012345.jpg") == "vegetales15"
    assert fuente_de("/kaggle/working/fotos_equipo/mango/web_mango_0001.jpg") == "web"
    assert fuente_de("/kaggle/working/fotos_equipo/mango/IMG_2024.jpg") == "drive"


def test_graficas_crea_los_png(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from src.graficas import graficas_datos, graficas_test

    filas, preds = [], []
    for i, (fuente, clase, etiqueta) in enumerate([("fruits360", "mango", "fruta"), ("drive", "col", "verdura"),
                                                   ("web", "mango", "fruta"), ("freshrotten", "col", "verdura")] * 4):
        ruta = tmp_path / f"{fuente}_{i}.jpg"
        Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(ruta)
        split = "test" if i % 2 else "train"
        filas.append([split, fuente, clase, etiqueta, ruta])
        if split == "test":
            preds.append([fuente, clase, etiqueta, "fruta", 0.4, ruta])
    for nombre, cabecera, datos in (("muestras.csv", ["split", "fuente", "clase", "etiqueta", "ruta"], filas),
                                    ("predicciones_test.csv",
                                     ["fuente", "clase", "real", "predicho", "prob_verdura", "ruta"], preds)):
        with open(tmp_path / nombre, "w", newline="") as f:
            csv.writer(f).writerows([cabecera, *datos])
    disponibles = [(str(r), 0 if e == "fruta" else 1, c) for _, _, c, e, r in filas]
    for png in graficas_datos(tmp_path, disponibles) + graficas_test(tmp_path, minimo=1):
        assert png.exists()
