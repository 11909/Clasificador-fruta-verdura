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

import joblib
import numpy as np
import yaml
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_validate

from src.dataset import conteo, limitar_por_clase, listar_muestras
from src.evaluate import metricas
from src.features import extraer_con_cache
from src.models import crear_modelo

METRICAS_CV = ["accuracy", "f1_macro"]


def crear_cv(cfg_cv: dict, semilla: int):
    """"grupos": ninguna clase (variedad) aparece a la vez en entrenamiento y validación, lo que evita
    que fotos casi idénticas de la misma pieza inflen el resultado. "estratificada": división aleatoria."""
    if cfg_cv["tipo"] == "grupos":
        return StratifiedGroupKFold(n_splits=cfg_cv["folds"], shuffle=True, random_state=semilla)
    if cfg_cv["tipo"] == "estratificada":
        return StratifiedKFold(n_splits=cfg_cv["folds"], shuffle=True, random_state=semilla)
    raise ValueError(f"cv.tipo desconocido: {cfg_cv['tipo']}")


def caracteristicas(muestras: list, cfg_feat: dict, cache_dir: Path, nombre: str) -> np.ndarray:
    rutas = [m[0] for m in muestras]
    clave = hashlib.md5(json.dumps([rutas, cfg_feat]).encode()).hexdigest()[:10]
    inicio = time.time()
    X = extraer_con_cache(rutas, cache_dir / f"{nombre}_{clave}.npy", **cfg_feat)
    print(f"características {nombre}: {X.shape} en {time.time() - inicio:.0f} s")
    return X


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    d, semilla = cfg["data"], cfg["seed"]
    salida = Path(cfg["output_dir"])
    salida.mkdir(parents=True, exist_ok=True)

    train = limitar_por_clase(listar_muestras(d["train_dirs"]), d.get("max_por_clase"), semilla)
    test = limitar_por_clase(listar_muestras(d["test_dirs"]), d.get("max_por_clase_test"), semilla) \
        if d.get("test_dirs") else []
    print(f"train {conteo(train)} | test {conteo(test)}")

    X = caracteristicas(train, cfg["features"], salida / "cache", "train")
    y = np.array([m[1] for m in train])
    grupos = np.array([m[2] for m in train])
    cv = crear_cv(cfg["cv"], semilla)

    resultados = []
    for spec in cfg["models"]:
        modelo = crear_modelo(spec["name"], spec.get("params"), spec.get("pca"), semilla)
        inicio = time.time()
        if spec.get("grid"):
            # Búsqueda de hiperparámetros con la misma validación cruzada
            busqueda = GridSearchCV(modelo, spec["grid"], cv=cv, scoring="f1_macro", n_jobs=-1)
            busqueda.fit(X, y, groups=grupos)
            fila_mejor = busqueda.best_index_
            puntajes = {"f1_macro": busqueda.cv_results_["mean_test_score"][fila_mejor],
                        "f1_macro_std": busqueda.cv_results_["std_test_score"][fila_mejor]}
            spec = {**spec, "mejores_params": busqueda.best_params_}
        else:
            cvr = cross_validate(modelo, X, y, groups=grupos, cv=cv, scoring=METRICAS_CV, n_jobs=-1)
            puntajes = {m: cvr[f"test_{m}"].mean() for m in METRICAS_CV}
            puntajes["f1_macro_std"] = cvr["test_f1_macro"].std()
        fila = {"modelo": spec["name"], **{k: round(float(v), 4) for k, v in puntajes.items()},
                "segundos": round(time.time() - inicio, 1), "spec": spec}
        print(json.dumps(fila, default=str))
        resultados.append(fila)

    with open(salida / "resultados_cv.csv", "w", newline="", encoding="utf-8") as f:
        campos = ["modelo", "accuracy", "f1_macro", "f1_macro_std", "segundos", "spec"]
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "spec": json.dumps(r["spec"], default=str)} for r in resultados)

    mejor = max(resultados, key=lambda r: r["f1_macro"])
    spec = mejor["spec"]
    print(f"mejor modelo en validación cruzada: {spec['name']} (f1_macro {mejor['f1_macro']})")
    modelo = crear_modelo(spec["name"], spec.get("params"), spec.get("pca"), semilla)
    if spec.get("mejores_params"):
        modelo.set_params(**spec["mejores_params"])
    modelo.fit(X, y)
    joblib.dump({"modelo": modelo, "features": cfg["features"], "spec": spec}, salida / "modelo.joblib")

    if test:
        X_test = caracteristicas(test, cfg["features"], salida / "cache", "test")
        resultado = metricas([m[1] for m in test], modelo.predict(X_test))
        print("test", json.dumps(resultado))
        json.dump({"modelo": spec["name"], **resultado}, open(salida / "test_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    main(parser.parse_args().config)
