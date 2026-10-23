import io

import numpy as np
import pytest
from PIL import Image

from src import modelo as modulo_modelo
from src.clasificador import Clasificador
from src.entrada import leer_imagen, listar_entradas, normalizar_url
from src.features import extraer
from src.modelo import cargar_modelo, guardar_modelo
from src.models import crear_modelo

FEATURES = {"tamano": 32, "tipos": ["color"], "bins_color": 8}


def imagen(color):
    return Image.new("RGB", (40, 40), color)


@pytest.fixture
def ruta_modelo(tmp_path):
    # rojo = fruta (0), verde = verdura (1)
    imgs = [imagen((200 + i, 10, 10)) for i in range(10)] + [imagen((10, 200 + i, 10)) for i in range(10)]
    X = np.vstack([extraer(i, **FEATURES) for i in imgs])
    y = np.array([0] * 10 + [1] * 10)
    modelo = crear_modelo("logistic_regression").fit(X, y)
    return guardar_modelo(modelo, FEATURES, {"name": "logistic_regression"}, tmp_path / "modelo.joblib")


def test_normalizar_url():
    assert normalizar_url("https://drive.google.com/file/d/ABC123/view?usp=sharing") == \
        "https://drive.google.com/uc?export=download&id=ABC123"
    assert normalizar_url("https://github.com/u/r/blob/main/fotos/a.jpg") == \
        "https://raw.githubusercontent.com/u/r/main/fotos/a.jpg"
    assert normalizar_url("https://ejemplo.com/a.jpg") == "https://ejemplo.com/a.jpg"


def test_listar_entradas_txt_y_carpeta(tmp_path):
    (tmp_path / "fotos").mkdir()
    imagen((1, 2, 3)).save(tmp_path / "fotos" / "a.jpg")
    imagen((1, 2, 3)).save(tmp_path / "fotos" / "b.png")
    (tmp_path / "fotos" / "notas.md").write_text("x")
    lista = tmp_path / "lista.txt"
    lista.write_text("# comentario\n\nhttps://ejemplo.com/x.jpg#frag\nfotos/a.jpg\n")
    assert listar_entradas(str(lista)) == ["https://ejemplo.com/x.jpg#frag", "fotos/a.jpg"]
    assert [p.split("/")[-1] for p in listar_entradas(str(tmp_path / "fotos"))] == ["a.jpg", "b.png"]
    assert listar_entradas(["una.jpg", "otra.jpg"]) == ["una.jpg", "otra.jpg"]


def test_leer_imagen_varios_tipos(tmp_path):
    img = imagen((5, 6, 7))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img.save(tmp_path / "a.png")
    for fuente in [img, buffer.getvalue(), np.asarray(img), str(tmp_path / "a.png")]:
        assert leer_imagen(fuente).mode == "RGB"


def test_clasificador_sin_reentrenar(ruta_modelo, tmp_path):
    clf = Clasificador(str(ruta_modelo))
    assert clf.clasificar(imagen((230, 20, 20)))["etiqueta"] == "fruta"
    assert clf.clasificar(imagen((20, 230, 20)))["etiqueta"] == "verdura"
    imagen((20, 230, 20)).save(tmp_path / "verde.jpg")
    resultados = clf.clasificar_lote([str(tmp_path / "verde.jpg"), str(tmp_path / "no_existe.jpg")])
    assert resultados[0]["etiqueta"] == "verdura"
    assert "error" in resultados[1]
    assert "logistic_regression" in clf.descripcion


def test_modelo_por_enlace_usa_copia_si_falla_la_descarga(ruta_modelo, tmp_path, monkeypatch):
    monkeypatch.setattr(modulo_modelo, "CACHE", tmp_path / "cache")
    url = "https://ejemplo.com/modelo.joblib"

    def descarga_ok(_url, destino, timeout=60):
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(ruta_modelo.read_bytes())
        return destino

    monkeypatch.setattr(modulo_modelo, "descargar", descarga_ok)
    assert cargar_modelo(url)["spec"]["name"] == "logistic_regression"

    def descarga_falla(*_args, **_kwargs):
        raise OSError("sin red")

    monkeypatch.setattr(modulo_modelo, "descargar", descarga_falla)
    with pytest.warns(UserWarning, match="copia guardada"):
        assert cargar_modelo(url)["spec"]["name"] == "logistic_regression"
    with pytest.raises(RuntimeError, match="Release"):
        cargar_modelo("https://ejemplo.com/otro.joblib")
