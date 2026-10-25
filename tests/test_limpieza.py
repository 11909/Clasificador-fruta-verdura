import numpy as np
from PIL import Image, ImageDraw

from src.limpieza import fondo_blanco, limpiar


def pieza(ruta, color, angulo, fondo=(255, 255, 255)):
    im = Image.new("RGB", (100, 100), fondo)
    d = ImageDraw.Draw(im)
    d.ellipse([25, 30, 75, 70], fill=color)
    d.line([50, 50, 50 + 20 * np.cos(angulo), 50 + 20 * np.sin(angulo)], fill=(0, 0, 0), width=4)
    im.save(ruta)
    return str(ruta)


def test_limpieza_quita_cuadros_repetidos_y_aplica_tope(tmp_path):
    # 30 cuadros casi iguales de Fruits-360 (la misma pieza apenas girada) y 5 fotos de otra fuente
    f360 = [(pieza(tmp_path / f"fruits360_{i}.jpg", (200, 30, 30), i * 0.01), 0, "apple") for i in range(30)]
    rng = np.random.default_rng(0)
    otras = []
    for i in range(5):
        ruta = tmp_path / f"packeat_{i}.jpg"
        Image.fromarray((rng.random((100, 100, 3)) * 255).astype("uint8")).save(ruta)
        otras.append((str(ruta), 0, "apple"))
    limpias = limpiar(f360 + otras, tope_fruits360=100, umbral_duplicado=6)
    assert len([m for m in limpias if "fruits360" in m[0]]) < 5
    assert all(m in limpias for m in otras)  # las otras fuentes no se tocan
    assert len(limpiar(f360, tope_fruits360=3, umbral_duplicado=None)) == 3


def test_fondo_blanco(tmp_path):
    assert fondo_blanco(pieza(tmp_path / "a.jpg", (200, 30, 30), 0))
    assert not fondo_blanco(pieza(tmp_path / "b.jpg", (200, 30, 30), 0, fondo=(90, 120, 60)))
