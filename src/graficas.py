"""Gráficas de los datos y de los errores en test para el reporte, a partir de los CSV que deja
src.train en la carpeta de salida (`muestras.csv` y `predicciones_test.csv`).

Uso desde el notebook:
    from src.graficas import graficas_datos, graficas_test
    graficas_datos("/kaggle/working/outputs", disponibles)   # disponibles: (ruta, etiqueta, clase) opcional
    graficas_test("/kaggle/working/outputs")
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.dataset import fuente_de
from src.mapping import ETIQUETAS

# Paleta categórica validada (orden fijo); cada fuente tiene siempre el mismo color
AZUL, NARANJA, AQUA, AMARILLO, MAGENTA, VERDE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
TEXTO, TEXTO_2, MUDO, REJILLA = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FUENTES = {"fruits360": ("Fruits-360", AZUL), "freshrotten": ("Fresh & Rotten", NARANJA),
           "packeat": ("PackEat", AQUA), "vegetales15": ("Vegetales 15", AMARILLO),
           "drive": ("Drive del equipo", MAGENTA), "web": ("Fotos web (FOTOS_WEB)", VERDE)}
SPLITS = {"train": AZUL, "test": NARANJA}
COLOR_ETIQUETA = {"fruta": AMARILLO, "verdura": VERDE}


def _estilo():
    plt.rcParams.update({"axes.edgecolor": REJILLA, "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2,
                         "ytick.color": TEXTO, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlelocation": "left", "axes.titlecolor": TEXTO, "axes.titlesize": 11,
                         "figure.facecolor": "white"})


def _ejes(ax, eje="x"):
    ax.grid(axis=eje, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y" if eje == "x" else "x", length=0)
    ax.spines["left" if eje == "x" else "bottom"].set_visible(False)


def _nombre(fuente):
    return FUENTES.get(fuente, (fuente, MUDO))[0]


def _color(fuente):
    return FUENTES.get(fuente, (fuente, MUDO))[1]


def _orden_fuentes(fuentes):
    """Las fuentes conocidas en su orden fijo y luego las demás."""
    fuentes = set(fuentes)
    return [f for f in FUENTES if f in fuentes] + sorted(fuentes - set(FUENTES))


def leer_csv(ruta) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _apiladas(ax, categorias, capas, conteos, colores, nombres, total=True, alto=0.6):
    """Barras horizontales apiladas: `conteos[(categoria, capa)]`, una capa por color."""
    y = np.arange(len(categorias))[::-1]
    izquierda = np.zeros(len(categorias))
    for capa in capas:
        v = np.array([conteos.get((c, capa), 0) for c in categorias], dtype=float)
        if v.sum():
            ax.barh(y, v, left=izquierda, height=alto, color=colores[capa], label=nombres[capa],
                    edgecolor="white", linewidth=1)
        izquierda += v
    if total:
        for yi, t in zip(y, izquierda):
            ax.text(t, yi, f" {int(t):,}", va="center", color=TEXTO_2, fontsize=8)
    ax.set_yticks(y, categorias)
    ax.set_xlim(0, izquierda.max() * 1.15 if izquierda.max() else 1)
    _ejes(ax)
    return izquierda


def graficas_datos(salida, disponibles: list | None = None) -> list[Path]:
    """Gráficas de los datos usados en la corrida (`muestras.csv`):

    - `datos_fuentes.png`: imágenes por fuente (disponibles y usadas), por split, por etiqueta y
      composición de train y test por fuente.
    - `datos_clases.png`: imágenes por alimento, separadas en fruta y verdura y apiladas por fuente.
    - `ejemplos_fuentes.png`: fotos de ejemplo de cada fuente.

    `disponibles`: muestras (ruta, etiqueta, clase) antes de limpiar y balancear, para comparar."""
    _estilo()
    salida = Path(salida)
    filas = leer_csv(salida / "muestras.csv")
    fuentes = _orden_fuentes(f["fuente"] for f in filas)
    nombres_fuente = {f: _nombre(f) for f in fuentes}
    colores_fuente = {f: _color(f) for f in fuentes}
    archivos = []

    # 1. Fuentes, splits y etiquetas
    por_fuente_split = Counter((nombres_fuente[f["fuente"]], f["split"]) for f in filas)
    por_etiqueta_split = Counter((f["etiqueta"], f["split"]) for f in filas)
    filas_fig = 3 if disponibles else 2
    fig, ejes = plt.subplots(filas_fig, 2, figsize=(14, 3.4 * filas_fig))
    ejes = ejes.ravel()
    k = 0
    if disponibles:
        ax = ejes[0]
        antes = Counter(fuente_de(m[0]) for m in disponibles)
        despues = Counter(f["fuente"] for f in filas)
        todas = _orden_fuentes(set(antes) | set(despues))
        y = np.arange(len(todas))[::-1]
        for desplazamiento, (cuenta, color, nombre) in zip((0.2, -0.2), ((antes, MUDO, "disponibles"),
                                                                          (despues, AZUL, "usadas"))):
            v = [cuenta.get(f, 0) for f in todas]
            ax.barh(y + desplazamiento, v, height=0.38, color=color, label=nombre)
            for yi, vi in zip(y, v):
                ax.text(vi, yi + desplazamiento, f" {vi:,}", va="center", color=TEXTO_2, fontsize=8)
        ax.set_yticks(y, [_nombre(f) for f in todas])
        ax.set_xlim(0, max(antes.values()) * 1.18)
        _ejes(ax)
        ax.legend(frameon=False, loc="lower right")
        ax.set_title(f"Imágenes disponibles y usadas por fuente\n{len(disponibles):,} disponibles → "
                     f"{len(filas):,} usadas (limpieza, topes por clase y balance)")
        ax = ejes[1]
        antes_et = Counter(ETIQUETAS[m[1]] for m in disponibles)
        despues_et = Counter(f["etiqueta"] for f in filas)
        y = np.arange(len(ETIQUETAS))[::-1]
        for desplazamiento, (cuenta, color, nombre) in zip((0.2, -0.2), ((antes_et, MUDO, "disponibles"),
                                                                          (despues_et, AZUL, "usadas"))):
            v = [cuenta.get(e, 0) for e in ETIQUETAS]
            ax.barh(y + desplazamiento, v, height=0.38, color=color, label=nombre)
            for yi, vi in zip(y, v):
                ax.text(vi, yi + desplazamiento, f" {vi:,}", va="center", color=TEXTO_2, fontsize=8)
        ax.set_yticks(y, ETIQUETAS)
        ax.set_xlim(0, max(antes_et.values()) * 1.18)
        _ejes(ax)
        ax.set_title("Fruta y verdura: disponibles y usadas")
        k = 2

    ax = ejes[k]
    _apiladas(ax, [nombres_fuente[f] for f in fuentes], list(SPLITS), por_fuente_split, SPLITS,
              {s: s for s in SPLITS})
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Imágenes usadas por fuente y split")

    ax = ejes[k + 1]
    _apiladas(ax, list(ETIQUETAS), list(SPLITS), por_etiqueta_split, SPLITS, {s: s for s in SPLITS})
    ax.legend(frameon=False, loc="lower right")
    n_clases = {e: len({f["clase"] for f in filas if f["etiqueta"] == e}) for e in ETIQUETAS}
    ax.set_title("Imágenes por etiqueta y split\n" +
                 ", ".join(f"{n} alimentos de {e}" for e, n in n_clases.items()))

    # Composición en porcentaje: qué fuente aporta cada parte de train y test, por etiqueta
    for ax, split in zip(ejes[k + 2:k + 4], SPLITS):
        del_split = [f for f in filas if f["split"] == split]
        conteos = Counter((f["etiqueta"], nombres_fuente[f["fuente"]]) for f in del_split)
        totales = Counter(f["etiqueta"] for f in del_split)
        pct = {c: 100 * n / totales[c[0]] for c, n in conteos.items()}
        y = np.arange(len(ETIQUETAS))[::-1]
        izquierda = np.zeros(len(ETIQUETAS))
        for f in fuentes:
            v = np.array([pct.get((e, nombres_fuente[f]), 0) for e in ETIQUETAS])
            ax.barh(y, v, left=izquierda, height=0.6, color=colores_fuente[f], label=nombres_fuente[f],
                    edgecolor="white", linewidth=1)
            for yi, l, vi in zip(y, izquierda, v):
                if vi >= 6:
                    ax.text(l + vi / 2, yi, f"{vi:.0f} %", ha="center", va="center", fontsize=8,
                            color=TEXTO if colores_fuente[f] == AMARILLO else "white")
            izquierda += v
        ax.set_yticks(y, [f"{e}\n({totales[e]:,})" for e in ETIQUETAS])
        ax.set_xlim(0, 100)
        ax.set_xlabel("% de las imágenes de la etiqueta")
        _ejes(ax)
        ax.set_title(f"Composición de {split} por fuente")
    ejes[k + 2].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.05, -0.32), ncol=len(fuentes))
    fig.tight_layout()
    archivos.append(salida / "datos_fuentes.png")
    fig.savefig(archivos[-1], dpi=150, bbox_inches="tight")
    plt.show()

    # 2. Imágenes por alimento, apiladas por fuente
    por_clase = Counter((f["clase"], f["fuente"]) for f in filas)
    clases = {e: sorted({f["clase"] for f in filas if f["etiqueta"] == e},
                        key=lambda c: -sum(por_clase.get((c, f), 0) for f in fuentes)) for e in ETIQUETAS}
    alto = max(len(v) for v in clases.values()) * 0.17 + 1.5
    fig, ejes = plt.subplots(1, 2, figsize=(14, alto))
    for ax, e in zip(ejes, ETIQUETAS):
        tot = _apiladas(ax, clases[e], fuentes, por_clase, colores_fuente, nombres_fuente, alto=0.75)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(-0.6, len(clases[e]) - 0.4)
        ax.set_xlabel("imágenes (train + test)")
        ax.set_title(f"{e.capitalize()}: {len(clases[e])} alimentos, {int(tot.sum()):,} imágenes\n"
                     f"mediana {np.median(tot):.0f} por alimento (mín. {tot.min():.0f}, máx. {tot.max():.0f})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colores_fuente[f]) for f in fuentes]
    fig.legend(handles, [nombres_fuente[f] for f in fuentes], loc="upper center", ncol=len(fuentes),
               frameon=False, bbox_to_anchor=(0.5, 1.0 + 0.4 / alto))
    fig.tight_layout()
    archivos.append(salida / "datos_clases.png")
    fig.savefig(archivos[-1], dpi=150, bbox_inches="tight")
    plt.show()

    # 3. Fotos de ejemplo de cada fuente
    archivos.append(ejemplos_por_fuente(filas, salida / "ejemplos_fuentes.png"))
    return archivos


def ejemplos_por_fuente(filas: list[dict], destino: Path, por_fuente: int = 6, semilla: int = 42) -> Path:
    """Una fila de fotos por fuente, de alimentos distintos, con su clase y etiqueta."""
    rng = random.Random(semilla)
    grupos = defaultdict(list)
    for f in filas:
        grupos[f["fuente"]].append(f)
    fuentes = _orden_fuentes(grupos)
    fig, ejes = plt.subplots(len(fuentes), por_fuente, figsize=(2 * por_fuente, 2.3 * len(fuentes)), squeeze=False)
    for fila_ejes, fuente in zip(ejes, fuentes):
        vistas, elegidas = set(), []
        for f in rng.sample(grupos[fuente], len(grupos[fuente])):
            if f["clase"] not in vistas:
                vistas.add(f["clase"])
                elegidas.append(f)
            if len(elegidas) == por_fuente:
                break
        for ax, f in zip(fila_ejes, elegidas + [None] * por_fuente):
            ax.axis("off")
            if f is None:
                continue
            try:
                img = Image.open(f["ruta"]).convert("RGB")
                img.thumbnail((256, 256))
                ax.imshow(img)
            except OSError:
                continue
            ax.set_title(f"{f['clase']}\n{f['etiqueta']}", fontsize=8, color=TEXTO_2, loc="center")
        fila_ejes[0].text(-0.08, 0.5, _nombre(fuente), transform=fila_ejes[0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=10, color=TEXTO)
    fig.tight_layout()
    fig.savefig(destino, dpi=120, bbox_inches="tight")
    plt.show()
    return destino


def graficas_test(salida, peores: int = 20, minimo: int = 5) -> list[Path]:
    """Errores del modelo elegido en test (`predicciones_test.csv`), en `errores_test.png`:
    accuracy por fuente y por etiqueta, los alimentos con más errores y la distribución de la
    probabilidad de verdura con el umbral elegido. Solo entran a los peores los alimentos con al
    menos `minimo` imágenes en test, para que una sola foto mal clasificada no los ponga hasta arriba."""
    _estilo()
    salida = Path(salida)
    if not (salida / "predicciones_test.csv").exists():
        print("No hay predicciones_test.csv (corrida sin test).")
        return []
    filas = leer_csv(salida / "predicciones_test.csv")
    umbral = 0.5
    if (salida / "test_metrics.json").exists():
        umbral = json.load(open(salida / "test_metrics.json")).get("umbral_verdura", 0.5)
    fig = plt.figure(figsize=(15, max(7, 0.28 * peores + 2)))
    rejilla = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
    ax_fuente, ax_prob, ax_peores = (fig.add_subplot(rejilla[0, 0]), fig.add_subplot(rejilla[1, 0]),
                                     fig.add_subplot(rejilla[:, 1]))

    # Accuracy en test por fuente, separada en fruta y verdura
    fuentes = _orden_fuentes(f["fuente"] for f in filas)
    y = np.arange(len(fuentes))[::-1]
    for desplazamiento, e in zip((0.2, -0.2), ETIQUETAS):
        acc, n = [], []
        for fuente in fuentes:
            grupo = [f for f in filas if f["fuente"] == fuente and f["real"] == e]
            n.append(len(grupo))
            acc.append(np.mean([f["real"] == f["predicho"] for f in grupo]) if grupo else np.nan)
        ax_fuente.barh(y + desplazamiento, np.nan_to_num(acc), height=0.38, color=COLOR_ETIQUETA[e], label=e)
        for yi, a, ni in zip(y, acc, n):
            if ni:
                ax_fuente.text(a, yi + desplazamiento, f" {a:.2f} (n={ni:,})", va="center", color=TEXTO_2, fontsize=8)
    ax_fuente.set_yticks(y, [_nombre(f) for f in fuentes])
    ax_fuente.set_xlim(0, 1.4)
    ax_fuente.set_xticks(np.linspace(0, 1, 6))
    ax_fuente.set_xlabel("accuracy en test (recall de la etiqueta)")
    _ejes(ax_fuente)
    ax_fuente.legend(frameon=False, loc="lower right")
    ax_fuente.set_title("Acierto en test por fuente y etiqueta")

    # Probabilidad de verdura según la etiqueta real
    if all(f["prob_verdura"] for f in filas):
        bins = np.linspace(0, 1, 26)
        for e in ETIQUETAS:
            p = [float(f["prob_verdura"]) for f in filas if f["real"] == e]
            ax_prob.hist(p, bins=bins, color=COLOR_ETIQUETA[e], alpha=0.6, label=f"{e} real (n={len(p):,})",
                         edgecolor="white", linewidth=1)
        ax_prob.axvline(umbral, color=TEXTO, linewidth=1.5, linestyle="--")
        ax_prob.text(umbral, ax_prob.get_ylim()[1], f" umbral {umbral:.2f}", color=TEXTO, fontsize=9, va="top")
        ax_prob.set_xlabel("probabilidad de verdura que da el modelo")
        ax_prob.set_ylabel("imágenes de test")
        _ejes(ax_prob, "y")
        ax_prob.spines["left"].set_visible(True)
        ax_prob.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        ax_prob.set_title("Qué tan seguro está el modelo (a la izquierda del umbral: fruta)")
    else:
        ax_prob.axis("off")

    # Alimentos con menor acierto
    por_clase = defaultdict(list)
    for f in filas:
        por_clase[(f["clase"], f["real"])].append(f["real"] == f["predicho"])
    peor = sorted([kv for kv in por_clase.items() if len(kv[1]) >= minimo], key=lambda kv: (np.mean(kv[1]), -len(kv[1])))[:peores]
    y = np.arange(len(peor))[::-1]
    acc = [np.mean(v) for _, v in peor]
    ax_peores.barh(y, acc, height=0.6, color=[COLOR_ETIQUETA[e] for (_, e), _ in peor])
    for yi, a, ((c, e), v) in zip(y, acc, peor):
        ax_peores.text(a, yi, f" {a:.2f} ({sum(v)}/{len(v)})", va="center", color=TEXTO_2, fontsize=8)
    ax_peores.set_yticks(y, [f"{c} ({e})" for (c, e), _ in peor], fontsize=8)
    ax_peores.set_xlim(0, 1.2)
    ax_peores.set_xticks(np.linspace(0, 1, 6))
    ax_peores.set_xlabel("accuracy en test")
    _ejes(ax_peores)
    ax_peores.legend([plt.Rectangle((0, 0), 1, 1, color=COLOR_ETIQUETA[e]) for e in ETIQUETAS], ETIQUETAS,
                     frameon=False, loc="lower right")
    ax_peores.set_title(f"Los {len(peor)} alimentos con menor acierto en test\n(con {minimo} o más imágenes en test)")
    fig.tight_layout()
    destino = salida / "errores_test.png"
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.show()
    return [destino]
