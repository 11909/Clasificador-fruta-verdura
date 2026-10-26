"""Entrenamiento con modelos clásicos: extrae características, compara modelos con validación
cruzada, reentrena el mejor con todo el conjunto de entrenamiento y lo evalúa en test.

Uso:  python -m src.train --config configs/baseline.yaml
"""
import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut, LeaveOneOut, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict, cross_validate)

from src.dataset import (balancear_por_etiqueta, completar_por_clase, conteo, dividir_equipo, es_web,
                         fuente_de, limitar_por_clase, listar_muestras, muestra_balanceada)
from src.evaluate import elegir_umbral, intervalos_bootstrap, metricas, predecir_con_umbral
from src.features import extraer_con_cache
from src.limpieza import limpiar
from src.mapping import ETIQUETAS
from src.modelo import guardar_modelo
from src.models import crear_modelo

# Métricas por modelo, en el orden de las tablas: globales y luego por clase (fruta, verdura)
ESCORES = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
           "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
           "recall_macro": "recall_macro", "f1_macro": "f1_macro",
           **{f"{nombre}_{e}": make_scorer(fn, pos_label=i, zero_division=0)
              for nombre, fn in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score))
              for i, e in enumerate(ETIQUETAS)}}
METRICAS_CV = list(ESCORES)


CV_DEJA_UNO = ("uno_fuera_grupo", "uno_fuera")


def crear_cv(cfg_cv: dict, semilla: int):
    """"grupos": ninguna clase (variedad) aparece a la vez en entrenamiento y validación, lo que evita
    que fotos casi idénticas de la misma pieza inflen el resultado. "estratificada": división aleatoria.
    "uno_fuera_grupo": leave-one-out por alimento, cada vuelta valida con una clase que el modelo no vio.
    "uno_fuera": leave-one-out clásico, una imagen por vuelta (tantos entrenamientos como imágenes).
    "anidada": `folds` vueltas por alimento para comparar modelos; dentro de cada una, la búsqueda de
    hiperparámetros usa otras `folds_internos` vueltas por alimento solo con los datos de entrenamiento."""
    if cfg_cv["tipo"] in ("grupos", "anidada"):
        return StratifiedGroupKFold(n_splits=cfg_cv["folds"], shuffle=True, random_state=semilla)
    if cfg_cv["tipo"] == "estratificada":
        return StratifiedKFold(n_splits=cfg_cv["folds"], shuffle=True, random_state=semilla)
    if cfg_cv["tipo"] == "uno_fuera_grupo":
        return LeaveOneGroupOut()
    if cfg_cv["tipo"] == "uno_fuera":
        return LeaveOneOut()
    raise ValueError(f"cv.tipo desconocido: {cfg_cv['tipo']}")


def evaluar_cv(modelo, X, y, grupos, cv, deja_uno: bool, params: dict | None = None) -> tuple[dict, dict]:
    """Métricas de validación cruzada (promedio y desviación entre vueltas) y lo que devolvió
    cross_validate. Con leave-one-out cada vuelta tiene una sola clase (o una sola imagen), así que se
    juntan las predicciones de todas las vueltas y se miden una vez (desviación 0).
    `params` va al fit de cada vuelta (p. ej. los grupos de la búsqueda interna en la validación anidada)."""
    if deja_uno:
        res = metricas(y, cross_val_predict(modelo, X, y, groups=grupos, cv=cv, n_jobs=-1))
        return {**{m: res[m] for m in METRICAS_CV}, **{f"{m}_std": 0.0 for m in METRICAS_CV}}, {}
    cvr = cross_validate(modelo, X, y, groups=grupos, cv=cv, scoring=ESCORES, n_jobs=-1, params=params,
                         return_estimator=params is not None)
    return ({**{m: cvr[f"test_{m}"].mean() for m in METRICAS_CV},
             **{f"{m}_std": cvr[f"test_{m}"].std() for m in METRICAS_CV}}, cvr)


def separar_vistos(grupos: np.ndarray, fraccion: float, semilla: int) -> np.ndarray:
    """Máscara de validación con `fraccion` de las fotos de cada alimento: valida con fotos nuevas de
    alimentos que el modelo sí vio, como pasa al usar la app con los alimentos del catálogo."""
    rng = np.random.default_rng(semilla)
    val = np.zeros(len(grupos), bool)
    for g in np.unique(grupos):
        idx = rng.permutation(np.flatnonzero(grupos == g))
        val[idx[:round(len(idx) * fraccion)]] = True
    return val


def caracteristicas(muestras: list, cfg_feat: dict, cache_dir: Path, nombre: str) -> np.ndarray:
    rutas = [m[0] for m in muestras]
    clave = hashlib.md5(json.dumps([rutas, cfg_feat]).encode()).hexdigest()[:10]
    inicio = time.time()
    X = extraer_con_cache(rutas, cache_dir / f"{nombre}_{clave}.npy", **cfg_feat)
    print(f"características {nombre}: {X.shape} en {time.time() - inicio:.0f} s")
    return X


def equipo(d: dict, excluir, semilla: int) -> tuple[list, list]:
    """Fotos del equipo: una parte fija a test, el resto a train."""
    dirs = d["equipo_dir"] if isinstance(d["equipo_dir"], list) else [d["equipo_dir"]]
    e_train, e_test = [], []
    for carpeta in dirs:
        a, b = dividir_equipo(carpeta, d.get("fraccion_test_equipo", 0.3), semilla, excluir)
        e_train, e_test = e_train + a, e_test + b
    if not d.get("fotos_web", True):
        e_train, e_test = [m for m in e_train if not es_web(m[0])], [m for m in e_test if not es_web(m[0])]
    print(f"fotos del equipo: train {conteo(e_train)} | test {conteo(e_test)}")
    return e_train, e_test


def kaggle(dirs: list, d: dict, excluir, semilla: int) -> list:
    """Imágenes del dataset combinado, con la limpieza de `data.limpieza` (null = sin limpiar)."""
    muestras = listar_muestras(dirs, excluir_grupos=excluir)
    return limpiar(muestras, semilla=semilla, **d["limpieza"]) if d.get("limpieza") else muestras


def splits_normales(d: dict, excluir, semilla: int) -> tuple[list, list]:
    """train/test del dataset combinado más las fotos del equipo. En train cada clase llega hasta su tope
    por clase: primero con las fotos de Drive y el resto con el dataset combinado."""
    e_train, e_test = equipo(d, excluir, semilla) if d.get("equipo_dir") else ([], [])
    train = completar_por_clase(e_train, kaggle(d["train_dirs"], d, excluir, semilla), d.get("max_por_clase"),
                                semilla, d.get("max_por_clase_verdura"))
    test = limitar_por_clase(kaggle(d["test_dirs"], d, excluir, semilla), d.get("max_por_clase_test"),
                             semilla) if d.get("test_dirs") else []
    test += e_test
    de_drive = set(e_train)
    n_drive = sum(1 for m in train if m in de_drive)
    print(f"train: {len(train)} imágenes, {n_drive} de Drive y {len(train) - n_drive} del dataset combinado")
    if d.get("balancear_etiquetas"):
        # Mismo número de frutas que de verduras al entrenar; la prueba se deja como está
        train = balancear_por_etiqueta(train, d.get("max_por_etiqueta"), semilla)
    return train, test


def muestra_preliminar(d: dict, excluir, semilla: int) -> tuple[list, list]:
    """Toma `muestra_por_etiqueta` frutas y las mismas verduras, repartidas por igual entre clases: cada
    clase toma primero sus fotos de Drive (con su división fija en train y test) y se completa con el
    dataset combinado, dividido por clase con `fraccion_test_muestra` a test."""
    todas = kaggle(d["train_dirs"] + (d.get("test_dirs") or []), d, excluir, semilla)
    e_train, e_test = equipo(d, excluir, semilla) if d.get("equipo_dir") else ([], [])
    print(f"muestra preliminar de {d['muestra_por_etiqueta']} por etiqueta, entre {conteo(e_train + e_test)} "
          f"fotos de Drive y {conteo(todas)} del dataset combinado")
    return muestra_balanceada(todas, d["muestra_por_etiqueta"], d.get("fraccion_test_muestra", 0.2), semilla,
                              (e_train, e_test))


def guardar_muestras(train: list, test: list, destino: Path):
    """Una fila por imagen usada (split, fuente, clase, etiqueta), para las gráficas del reporte."""
    with open(destino, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "fuente", "clase", "etiqueta", "ruta"])
        for split, muestras in (("train", train), ("test", test)):
            writer.writerows([split, fuente_de(r), c, ETIQUETAS[y], r] for r, y, c in muestras)


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    d, semilla = cfg["data"], cfg["seed"]
    salida = Path(cfg["output_dir"])
    salida.mkdir(parents=True, exist_ok=True)

    excluir = d.get("excluir_grupos") or []
    if d.get("muestra_por_etiqueta"):
        train, test = muestra_preliminar(d, excluir, semilla)
    else:
        train, test = splits_normales(d, excluir, semilla)
    print(f"train {conteo(train)} | test {conteo(test)}")
    guardar_muestras(train, test, salida / "muestras.csv")

    X = caracteristicas(train, cfg["features"], salida / "cache", "train")
    y = np.array([m[1] for m in train])
    grupos = np.array([m[2] for m in train])
    cv = crear_cv(cfg["cv"], semilla)
    deja_uno = cfg["cv"]["tipo"] in CV_DEJA_UNO
    # GridSearchCV promedia un puntaje por vuelta, que no tiene sentido con una clase por vuelta:
    # la búsqueda usa 5 vueltas por grupos y el modelo elegido se mide después con leave-one-out
    cv_grid = StratifiedGroupKFold(n_splits=cfg["cv"].get("folds", 5), shuffle=True, random_state=semilla) \
        if deja_uno else cv
    anidada = cfg["cv"]["tipo"] == "anidada"
    if anidada:
        # La búsqueda de hiperparámetros va dentro de cada vuelta externa, con sus propias vueltas por alimento
        cv_grid = StratifiedGroupKFold(n_splits=cfg["cv"].get("folds_internos", 3), shuffle=True,
                                       random_state=semilla)

    X_test = caracteristicas(test, cfg["features"], salida / "cache", "test") if test else None
    y_test = [m[1] for m in test]
    grupos_test = [m[2] for m in test]

    # Criterio para elegir el modelo final: "vistos" (f1 con fotos nuevas de alimentos vistos) o "cv"
    elegir_por = cfg.get("elegir_por", "cv")
    val = separar_vistos(grupos, cfg.get("fraccion_vistos", 0.2), semilla) if elegir_por == "vistos" else None

    resultados, en_test = [], []
    # Escalado y PCA ya ajustados, compartidos entre combinaciones del grid y entre modelos con el mismo PCA
    cache_pca = tempfile.mkdtemp(prefix="pca_")
    for spec in cfg["models"]:
        if not cfg.get("ajustar_hiperparametros", True):
            spec = {k: v for k, v in spec.items() if k != "grid"}  # usa solo los params fijos
        modelo = crear_modelo(spec["name"], spec.get("params"), spec.get("pca"), semilla, cache=cache_pca)
        inicio = time.time()
        if spec.get("grid") and anidada:
            # Validación anidada: cada vuelta externa busca sus hiperparámetros solo con su parte de
            # entrenamiento, así la métrica de la vuelta externa no está inflada por la búsqueda
            interna = GridSearchCV(modelo, spec["grid"], cv=cv_grid, scoring="f1_macro", n_jobs=1)
            puntajes, cvr = evaluar_cv(interna, X, y, grupos, cv, False, params={"groups": grupos})
            print(f"{spec['name']}: hiperparámetros elegidos en cada vuelta externa: "
                  f"{[e.best_params_ for e in cvr['estimator']]}")
            # Para el modelo final, la misma búsqueda con todo el entrenamiento
            busqueda = GridSearchCV(modelo, spec["grid"], cv=cv_grid, scoring="f1_macro", n_jobs=-1)
            busqueda.fit(X, y, groups=grupos)
            spec = {**spec, "mejores_params": busqueda.best_params_}
            modelo = busqueda.best_estimator_
            print(f"{spec['name']}: hiperparámetros finales {busqueda.best_params_}")
        else:
            if spec.get("grid"):
                busqueda = GridSearchCV(modelo, spec["grid"], cv=cv_grid, scoring="f1_macro", n_jobs=-1)
                busqueda.fit(X, y, groups=grupos)
                spec = {**spec, "mejores_params": busqueda.best_params_}
                print(f"{spec['name']}: mejores hiperparámetros {busqueda.best_params_} "
                      f"(f1_macro {busqueda.best_score_:.4f} en la búsqueda)")
                modelo = busqueda.best_estimator_
            # Todas las métricas con la misma validación, también para los modelos con grid
            puntajes, _ = evaluar_cv(modelo, X, y, grupos, cv, deja_uno)
        if val is not None:
            vistos = clone(modelo).fit(X[~val], y[~val])
            puntajes["f1_macro_vistos"] = f1_score(y[val], vistos.predict(X[val]), average="macro")
        fila = {"modelo": spec["name"], **{k: round(float(v), 4) for k, v in puntajes.items()},
                "segundos": round(time.time() - inicio, 1), "spec": spec}
        print(json.dumps(fila, default=str))
        resultados.append(fila)
        if test:
            # Cada modelo entrenado con todo train y medido en test con umbral 0.5, para compararlos
            predichos_test = modelo.fit(X, y).predict(X_test)
            res = metricas(y_test, predichos_test)
            if cfg.get("bootstrap"):
                res.update(intervalos_bootstrap(y_test, predichos_test, grupos_test, cfg["bootstrap"], semilla))
            en_test.append({"modelo": spec["name"], **{k: round(v, 4) if isinstance(v, float) else v
                                                       for k, v in res.items()}})
            print("test", json.dumps(en_test[-1]))
    shutil.rmtree(cache_pca, ignore_errors=True)

    campos = ["modelo", *METRICAS_CV, *(f"{m}_std" for m in METRICAS_CV),
              *(["f1_macro_vistos"] if val is not None else []), "segundos", "spec"]
    with open(salida / "resultados_cv.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "spec": json.dumps(r["spec"], default=str)} for r in resultados)
    if en_test:
        with open(salida / "resultados_test.csv", "w", newline="", encoding="utf-8") as f:
            intervalos = [k for k in en_test[0] if "_ic_" in k]
            writer = csv.DictWriter(f, fieldnames=["modelo", *METRICAS_CV, *intervalos, "matriz_confusion"])
            writer.writeheader()
            writer.writerows({**r, "matriz_confusion": json.dumps(r["matriz_confusion"])} for r in en_test)

    criterio = "f1_macro_vistos" if val is not None else "f1_macro"
    mejor = max(resultados, key=lambda r: r[criterio])
    spec = mejor["spec"]
    print(f"mejor modelo por {criterio}: {spec['name']} ({mejor[criterio]})")
    modelo = crear_modelo(spec["name"], spec.get("params"), spec.get("pca"), semilla)
    if spec.get("mejores_params"):
        modelo.set_params(**spec["mejores_params"])
    umbral = 0.5
    if cfg.get("ajustar_umbral") and hasattr(modelo, "predict_proba"):
        # Probabilidades de validación cruzada (cada imagen predicha por un modelo que no la vio)
        probs = cross_val_predict(modelo, X, y, groups=grupos, cv=cv, method="predict_proba", n_jobs=-1)
        umbral = elegir_umbral(y, probs)
        print(f"umbral de verdura: {umbral} (accuracy balanceada en validación cruzada: "
              f"{metricas(y, predecir_con_umbral(probs, umbral))['balanced_accuracy']:.4f} "
              f"vs {metricas(y, predecir_con_umbral(probs, 0.5))['balanced_accuracy']:.4f} con 0.5)")
    modelo.fit(X, y)
    guardar_modelo(modelo, cfg["features"], spec, salida / "modelo.joblib", umbral=umbral)

    if test:
        probs = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None
        predichos = (predecir_con_umbral(np.c_[1 - probs, probs], umbral) if probs is not None
                     else modelo.predict(X_test))
        with open(salida / "predicciones_test.csv", "w", newline="", encoding="utf-8") as f:
            # Predicción del modelo elegido en cada imagen de test, para ver los errores por fuente y alimento
            writer = csv.writer(f)
            writer.writerow(["fuente", "clase", "real", "predicho", "prob_verdura", "ruta"])
            writer.writerows([fuente_de(m[0]), m[2], ETIQUETAS[m[1]], ETIQUETAS[p],
                              "" if probs is None else round(float(q), 4), m[0]]
                             for m, p, q in zip(test, predichos, probs if probs is not None else [None] * len(test)))
        resultado = metricas(y_test, predichos)
        print("test", json.dumps(resultado))
        json.dump({"modelo": spec["name"], "umbral_verdura": umbral, **resultado}, open(salida / "test_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    main(parser.parse_args().config)
