"""Clasifica imágenes desde la línea de comandos sin reentrenar.

Uso:
    python -m src.predict foto.jpg
    python -m src.predict https://ejemplo.com/foto.jpg otra.jpg
    python -m src.predict entradas/ejemplos.txt              # una ruta o enlace por línea
    python -m src.predict carpeta/ --modelo outputs/modelo.joblib

Sin --modelo se usa el modelo publicado en el Release más reciente del repo.
"""
import argparse

from src.clasificador import Clasificador


def cargar(ruta_modelo: str):
    """Compatibilidad con código anterior: devuelve un Clasificador."""
    return Clasificador(ruta_modelo)


def predecir(clasificador: Clasificador, imagen) -> dict[str, float]:
    """Compatibilidad con código anterior: probabilidades de fruta y verdura."""
    return clasificador.probabilidades(imagen)


def main():
    parser = argparse.ArgumentParser(description="Clasifica imágenes como fruta o verdura.")
    parser.add_argument("entradas", nargs="+", help="rutas, enlaces, carpetas o archivos .txt")
    parser.add_argument("--modelo", help="ruta o enlace del modelo (por defecto, el publicado)")
    args = parser.parse_args()

    clf = Clasificador(args.modelo)
    print(f"modelo: {clf.descripcion}\n")
    for r in clf.clasificar_lote(args.entradas):
        if "error" in r:
            print(f"ERROR    {r['entrada']}: {r['error']}")
        else:
            print(f"{r['etiqueta']:8} {r['confianza']:.0%}  {r['entrada']}")


if __name__ == "__main__":
    main()
