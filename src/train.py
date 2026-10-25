"""Entrenamiento con modelos clásicos: extrae características, compara modelos con validación
cruzada, reentrena el mejor con todo el conjunto de entrenamiento y lo evalúa en test.

Uso:  python -m src.train --config configs/baseline.yaml
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut, LeaveOneOut, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict, cross_validate)

from src.dataset import (balancear_por_etiqueta, conteo, dividir_equipo, limitar_por_clase, listar_muestras,
                         muestra_balanceada)
from src.evaluate import elegir_umbral, metricas, predecir_con_umbral
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
    "uno_fuera": leave-one-out clásico, una imagen por vuelta (tantos entrenamientos como imágenes)."""
    if cfg_cv["tipo"] == "grupos":
        return StratifiedGroupKFold(n_splits=cfg_cv["folds"], shuffle=True, random_state=semilla)
    if cfg_cv["tipo"] == "estratificada":
        return StratifiedKFold(n_splits=cfg_cv["folds"], shuffle=True, random_state=semilla)
    if cfg_cv["tipo"] == "uno_fuera_grupo":
        return LeaveOneGroupOut()
    if cfg_cv["tipo"] == "uno_fuera":
        return LeaveOneOut()
    raise ValueError(f"cv.tipo desconocido: {cfg_cv['tipo']}")


def evaluar_cv(modelo, X, y, grupos, cv, deja_uno: bool) -> dict:
    """Métricas de validación cruzada. Con leave-one-out cada vuelta tiene una sola clase (o una sola
    imagen), así que se juntan las predicciones de todas las vueltas y se miden una vez."""
    if deja_uno:
        res = metricas(y, cross_val_predict(modelo, X, y, groups=grupos, cv=cv, n_jobs=-1))
        return {**{m: res[m] for m in METRICAS_CV}, "f1_macro_std": 0.0}
    cvr = cross_validate(modelo, X, y, groups=grupos, cv=cv, scoring=ESCORES, n_jobs=-1)
    return {**{m: cvr[f"test_{m}"].mean() for m in METRICAS_CV}, "f1_macro_std": cvr["test_f1_macro"].std()}


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
    print(f"fotos del equipo: train {conteo(e_train)} | test {conteo(e_test)}")
    return e_train, e_test


def kaggle(dirs: list, d: dict, excluir, semilla: int) -> list:
    """Imágenes del dataset combinado, con la limpieza de `data.limpieza` (null = sin limpiar)."""
    muestras = listar_muestras(dirs, excluir_grupos=excluir)
    return limpiar(muestras, semilla=semilla, **d["limpieza"]) if d.get("limpieza") else muestras


def splits_normales(d: dict, excluir, semilla: int) -> tuple[list, list]:
    """train/test del dataset combinado (con tope por clase) más las fotos del equipo."""
    train = limitar_por_clase(kaggle(d["train_dirs"], d, excluir, semilla), d.get("max_por_clase"), semilla,
                              d.get("max_por_clase_verdura"))
    test = limitar_por_clase(kaggle(d["test_dirs"], d, excluir, semilla), d.get("max_por_clase_test"),
                             semilla) if d.get("test_dirs") else []
    if d.get("equipo_dir"):
        e_train, e_test = equipo(d, excluir, semilla)
        train, test = train + e_train, test + e_test
    if d.get("balancear_etiquetas"):
        # Mismo número de frutas que de verduras al entrenar; la prueba se deja como está
        train = balancear_por_etiqueta(train, d.get("max_por_etiqueta"), semilla)
    return train, test


def muestra_preliminar(d: dict, excluir, semilla: int) -> tuple[list, list]:
    """Toma `muestra_por_etiqueta` frutas y las mismas verduras: todas las fotos del equipo (con su
    división fija en train y test) y el resto del dataset combinado, dividido por clase con
    `fraccion_test_muestra` a test."""
    todas = kaggle(d["train_dirs"] + (d.get("test_dirs") or []), d, excluir, semilla)
    e_train, e_test = equipo(d, excluir, semilla) if d.get("equipo_dir") else ([], [])
    print(f"muestra preliminar de {d['muestra_por_etiqueta']} por etiqueta: todas las del equipo "
          f"{conteo(e_train + e_test)} y el resto de {conteo(todas)} del dataset combinado")
    return muestra_balanceada(todas, d["muestra_por_etiqueta"], d.get("fraccion_test_muestra", 0.2), semilla,
                              (e_train, e_test))


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

    X = caracteristicas(train, cfg["features"], salida / "cache", "train")
    y = np.array([m[1] for m in train])
    grupos = np.array([m[2] for m in train])
    cv = crear_cv(cfg["cv"], semilla)
    deja_uno = cfg["cv"]["tipo"] in CV_DEJA_UNO
    # GridSearchCV promedia un puntaje por vuelta, que no tiene sentido con una clase por vuelta:
    # la búsqueda usa 5 vueltas por grupos y el modelo elegido se mide después con leave-one-out
    cv_grid = StratifiedGroupKFold(n_splits=cfg["cv"].get("folds", 5), shuffle=True, random_state=semilla) \
        if deja_uno else cv

    X_test = caracteristicas(test, cfg["features"], salida / "cache", "test") if test else None
    y_test = [m[1] for m in test]

    resultados, en_test = [], []
    for spec in cfg["models"]:
        modelo = crear_modelo(spec["name"], spec.get("params"), spec.get("pca"), semilla)
        inicio = time.time()
        if spec.get("grid"):
            busqueda = GridSearchCV(modelo, spec["grid"], cv=cv_grid, scoring="f1_macro", n_jobs=-1)
            busqueda.fit(X, y, groups=grupos)
            spec = {**spec, "mejores_params": busqueda.best_params_}
            modelo = busqueda.best_estimator_
        # Todas las métricas con la misma validación, también para los modelos con grid
        puntajes = evaluar_cv(modelo, X, y, grupos, cv, deja_uno)
        fila = {"modelo": spec["name"], **{k: round(float(v), 4) for k, v in puntajes.items()},
                "segundos": round(time.time() - inicio, 1), "spec": spec}
        print(json.dumps(fila, default=str))
        resultados.append(fila)
        if test:
            # Cada modelo entrenado con todo train y medido en test con umbral 0.5, para compararlos
            res = metricas(y_test, modelo.fit(X, y).predict(X_test))
            en_test.append({"modelo": spec["name"], **{k: round(v, 4) if isinstance(v, float) else v
                                                       for k, v in res.items()}})
            print("test", json.dumps(en_test[-1]))

    campos = ["modelo", *METRICAS_CV, "f1_macro_std", "segundos", "spec"]
    with open(salida / "resultados_cv.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "spec": json.dumps(r["spec"], default=str)} for r in resultados)
    if en_test:
        with open(salida / "resultados_test.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["modelo", *METRICAS_CV, "matriz_confusion"])
            writer.writeheader()
            writer.writerows({**r, "matriz_confusion": json.dumps(r["matriz_confusion"])} for r in en_test)

    mejor = max(resultados, key=lambda r: r["f1_macro"])
    spec = mejor["spec"]
    print(f"mejor modelo en validación cruzada: {spec['name']} (f1_macro {mejor['f1_macro']})")
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
        predichos = (predecir_con_umbral(modelo.predict_proba(X_test), umbral)
                     if hasattr(modelo, "predict_proba") else modelo.predict(X_test))
        resultado = metricas(y_test, predichos)
        print("test", json.dumps(resultado))
        json.dump({"modelo": spec["name"], "umbral_verdura": umbral, **resultado}, open(salida / "test_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    main(parser.parse_args().config)
