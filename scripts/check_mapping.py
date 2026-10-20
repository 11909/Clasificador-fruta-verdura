"""Muestra qué clases de un dataset faltan en data/mapping.csv y cómo queda cada una.

Uso:  python scripts/check_mapping.py /ruta/al/dataset/Training [otra/ruta ...]
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mapping import cargar_mapping, etiqueta_de  # noqa: E402

mapping = cargar_mapping()
resumen = Counter()
for raiz in sys.argv[1:]:
    for carpeta in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
        etiqueta = etiqueta_de(carpeta.name, mapping) or "SIN MAPEO"
        resumen[etiqueta] += 1
        print(f"{etiqueta:10} {carpeta.name}")
print("\nClases por etiqueta:", dict(resumen))
sys.exit(1 if resumen["SIN MAPEO"] else 0)
