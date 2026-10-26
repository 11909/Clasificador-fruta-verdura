"""Lectura de carpetas tipo ImageFolder (raiz/clase/imagen.jpg) con etiqueta binaria."""
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.mapping import ETIQUETAS, EXCLUIR, cargar_alimentos, cargar_mapping, clave_de, etiqueta_de, normalizar

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_muestras(raices: list[str], mapping: dict[str, str] | None = None,
                    excluir_grupos=()) -> list[tuple[str, int, str]]:
    """Recorre cada raíz y devuelve (ruta, etiqueta, clase) con etiqueta 0 = fruta y 1 = verdura.

    Falla si alguna carpeta de clase no está en data/mapping.csv, para no entrenar con etiquetas faltantes.
    """
    mapping = mapping or cargar_mapping(excluir_grupos=excluir_grupos)
    muestras, sin_mapeo = [], set()
    for raiz in raices:
        for carpeta in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
            etiqueta = etiqueta_de(carpeta.name, mapping)
            if etiqueta is None:
                sin_mapeo.add(carpeta.name)
                continue
            if etiqueta == EXCLUIR:
                continue
            y, clase = ETIQUETAS.index(etiqueta), normalizar(carpeta.name)
            muestras += [(str(p), y, clase) for p in sorted(carpeta.iterdir()) if p.suffix.lower() in EXTENSIONES]
    if sin_mapeo:
        raise ValueError(f"Clases sin entrada en data/mapping.csv: {sorted(sin_mapeo)}")
    return muestras


def va_a_test(ruta, fraccion_test: float, semilla: int) -> bool:
    """Decide por hash de la carpeta y el nombre del archivo si una foto va a test, de modo que la
    asignación no cambia al agregar fotos nuevas ni entre versiones del dataset."""
    ruta = Path(ruta)
    clave = f"{semilla}/{ruta.parent.name}/{ruta.name}".encode()
    return int(hashlib.md5(clave).hexdigest(), 16) % 1000 < fraccion_test * 1000


def dividir_equipo(raiz: str, fraccion_test: float, semilla: int, excluir_grupos=()) -> tuple[list, list]:
    """Fotos del equipo (raiz/alimento/foto.jpg) repartidas entre train y test con va_a_test."""
    raiz = Path(raiz)
    if not raiz.is_dir():
        return [], []
    # Si el zip se subió con una carpeta envolvente (fotos/zanahoria/...), se entra en ella
    mapping = cargar_mapping(excluir_grupos=excluir_grupos)
    while True:
        subcarpetas = [p for p in raiz.iterdir() if p.is_dir()]
        if len(subcarpetas) != 1 or etiqueta_de(subcarpetas[0].name, mapping) is not None:
            break
        raiz = subcarpetas[0]
    muestras = listar_muestras([str(raiz)], mapping)
    test = [m for m in muestras if va_a_test(m[0], fraccion_test, semilla)]
    en_test = {m[0] for m in test}
    return [m for m in muestras if m[0] not in en_test], test


def limitar_por_clase(muestras: list, maximo: int | None, semilla: int, maximo_verdura: int | None = None) -> list:
    """Toma como máximo `maximo` imágenes de cada clase, para acotar memoria y tiempo. Las clases de
    verdura usan `maximo_verdura` si se da, para compensar que hay menos clases de verdura."""
    if not maximo and not maximo_verdura:
        return muestras
    rng = random.Random(semilla)
    por_clase = defaultdict(list)
    for m in muestras:
        por_clase[m[2]].append(m)
    salida = []
    for grupo in por_clase.values():
        tope = (maximo_verdura or maximo) if grupo[0][1] == ETIQUETAS.index("verdura") else maximo
        salida += rng.sample(grupo, min(tope, len(grupo))) if tope else grupo
    return salida


def repartir(muestras: list, n: int, rng: random.Random) -> list:
    """Toma `n` imágenes repartidas por igual entre clases (una de cada clase por turno)."""
    clases = defaultdict(list)
    for m in muestras:
        clases[m[2]].append(m)
    colas = [rng.sample(grupo, len(grupo)) for _, grupo in sorted(clases.items())]
    rng.shuffle(colas)
    tomadas = []
    while len(tomadas) < n and any(colas):
        for cola in colas:
            if cola and len(tomadas) < n:
                tomadas.append(cola.pop())
    return tomadas


def balancear_por_etiqueta(muestras: list, maximo: int | None, semilla: int) -> list:
    """Deja el mismo número de imágenes de fruta y de verdura: el de la etiqueta con menos
    imágenes, o `maximo` si es menor. Dentro de cada etiqueta reparte por igual entre clases
    (una imagen de cada clase por turno) para no perder variedades."""
    rng = random.Random(semilla)
    por_etiqueta = defaultdict(list)
    for m in muestras:
        por_etiqueta[m[1]].append(m)
    if len(por_etiqueta) < 2:
        return muestras
    objetivo = min(len(v) for v in por_etiqueta.values())
    if maximo:
        objetivo = min(objetivo, maximo)
    return [m for _, grupo in sorted(por_etiqueta.items()) for m in repartir(grupo, objetivo, rng)]


def es_web(ruta) -> bool:
    """Fotos de internet de scripts/web_scrapping.py (web_<alimento>_0001.jpg), que van en Drive con las del equipo."""
    return Path(ruta).name.startswith("web_")


def _colas_por_clase(primero: list, despues: list, rng: random.Random) -> dict[str, list]:
    """Por clase, las fotos de `primero` (Drive) y luego las de `despues` (dataset combinado), cada
    grupo barajado: así una clase toma primero las fotos reales y se completa con las demás. Las clases
    se juntan por alimento (`rabano_negro` de Drive cuenta como `rabano`)."""
    alimentos, claves = cargar_alimentos(), {}
    a, b = defaultdict(list), defaultdict(list)
    for destino, grupo in ((a, primero), (b, despues)):
        for m in grupo:
            if m[2] not in claves:
                clave = clave_de(m[2], alimentos)
                claves[m[2]] = alimentos[clave] if clave else m[2]
            destino[claves[m[2]]].append(m)
    return {c: rng.sample(a[c], len(a[c])) + rng.sample(b[c], len(b[c])) for c in sorted(set(a) | set(b))}


def completar_por_clase(equipo: list, kaggle: list, maximo: int | None, semilla: int,
                        maximo_verdura: int | None = None) -> list:
    """Cada clase entrena con hasta `maximo` imágenes (`maximo_verdura` en las verduras) entre Drive y
    el dataset combinado: primero las de Drive y el resto del combinado, para que una clase con muchas
    fotos en Drive no quede más grande que las demás."""
    if not maximo and not maximo_verdura:
        return equipo + kaggle
    salida = []
    for cola in _colas_por_clase(equipo, kaggle, random.Random(semilla)).values():
        tope = (maximo_verdura or maximo) if cola[0][1] == ETIQUETAS.index("verdura") else maximo
        salida += cola[:tope] if tope else cola
    return salida


def muestra_balanceada(muestras: list, por_etiqueta: int, fraccion_test: float, semilla: int,
                       equipo: tuple[list, list] = ([], [])) -> tuple[list, list]:
    """Toma `por_etiqueta` imágenes de fruta y las mismas de verdura y las divide en train y test.

    Dentro de cada etiqueta reparte por igual entre clases (una imagen de cada clase por turno). Cada
    clase toma primero sus fotos de Drive (`equipo` = (train, test), con su división fija) y se completa
    con `muestras`; de esas, `fraccion_test` de cada clase va a test. Si a una etiqueta no le alcanza,
    las dos se quedan con lo que tenga la más corta."""
    rng = random.Random(semilla)
    e_train, e_test = equipo
    del_equipo, en_test = {m[0] for m in e_train + e_test}, {m[0] for m in e_test}
    colas = defaultdict(list)
    for clase, cola in _colas_por_clase(e_train + e_test, muestras, rng).items():
        colas[cola[0][1]].append(cola[::-1])  # se toma del final
    disponibles = [sum(len(c) for c in colas[y]) for y in range(len(ETIQUETAS))]
    por_etiqueta = min(por_etiqueta, *disponibles)
    train, test, por_clase = [], [], defaultdict(list)
    for y in range(len(ETIQUETAS)):
        rng.shuffle(colas[y])
        tomadas = 0
        while tomadas < por_etiqueta:
            for cola in colas[y]:
                if cola and tomadas < por_etiqueta:
                    m = cola.pop()
                    tomadas += 1
                    if m[0] in del_equipo:
                        (test if m[0] in en_test else train).append(m)
                    else:
                        por_clase[m[2]].append(m)
    for _, grupo in sorted(por_clase.items()):
        rng.shuffle(grupo)
        n_test = round(len(grupo) * fraccion_test)
        test += grupo[:n_test]
        train += grupo[n_test:]
    return train, test


def conteo(muestras: list) -> dict[str, int]:
    c = Counter(m[1] for m in muestras)
    return {ETIQUETAS[y]: c.get(y, 0) for y in range(len(ETIQUETAS))}
