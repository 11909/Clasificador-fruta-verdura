"""Entrenamiento con transfer learning.

Uso:  python -m src.train --config configs/baseline.yaml
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import ImagenesBinarias, conteo, dividir, listar_muestras, transformaciones
from src.evaluate import evaluar
from src.mapping import ETIQUETAS
from src.model import congelar_backbone, crear_modelo


def fijar_semilla(semilla: int):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def submuestra(muestras: list, maximo: int | None, semilla: int) -> list:
    """Para pruebas rápidas: limita el número de imágenes."""
    if not maximo or len(muestras) <= maximo:
        return muestras
    return random.Random(semilla).sample(muestras, maximo)


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    fijar_semilla(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    salida = Path(cfg["output_dir"])
    salida.mkdir(parents=True, exist_ok=True)

    muestras = submuestra(listar_muestras(d["train_dirs"]), d.get("max_samples"), t["seed"])
    train, val = dividir(muestras, d["val_fraction"], t["seed"])
    test = submuestra(listar_muestras(d["test_dirs"]), d.get("max_samples"), t["seed"]) if d.get("test_dirs") else []
    print(f"train {conteo(train)} | val {conteo(val)} | test {conteo(test)} | device {device}")

    cargar = lambda ms, ent: DataLoader(
        ImagenesBinarias(ms, transformaciones(d["image_size"], ent)),
        batch_size=t["batch_size"], shuffle=ent, num_workers=t["num_workers"], pin_memory=device == "cuda",
    )
    dl_train, dl_val = cargar(train, True), cargar(val, False)

    modelo = crear_modelo(m["name"], m["pretrained"]).to(device)
    # Pesos inversos a la frecuencia para compensar el desbalance fruta/verdura
    n = conteo(train)
    pesos = torch.tensor([len(train) / (len(ETIQUETAS) * max(n[e], 1)) for e in ETIQUETAS], device=device)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    optim = torch.optim.AdamW(modelo.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t["epochs"])
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    wandb_run = None
    if cfg.get("wandb", {}).get("enabled"):
        import wandb
        wandb_run = wandb.init(project=cfg["wandb"]["project"], config=cfg)

    mejor_f1 = -1.0
    for epoca in range(t["epochs"]):
        congelar_backbone(modelo, epoca < t["freeze_backbone_epochs"])
        modelo.train()
        perdida_total = 0.0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                perdida = criterio(modelo(x), y)
            scaler.scale(perdida).backward()
            scaler.step(optim)
            scaler.update()
            perdida_total += perdida.item() * len(y)
        sched.step()

        metricas = evaluar(modelo, dl_val, device)
        registro = {"epoca": epoca + 1, "loss_train": perdida_total / len(train),
                    **{f"val_{k}": v for k, v in metricas.items() if k != "matriz_confusion"}}
        print(json.dumps(registro))
        if wandb_run:
            wandb_run.log(registro)
        if metricas["f1_macro"] > mejor_f1:
            mejor_f1 = metricas["f1_macro"]
            torch.save({"model": m["name"], "image_size": d["image_size"], "state_dict": modelo.state_dict()},
                       salida / "best.pt")

    if test:
        modelo.load_state_dict(torch.load(salida / "best.pt", map_location=device)["state_dict"])
        metricas = evaluar(modelo, cargar(test, False), device)
        print("test", json.dumps(metricas))
        json.dump(metricas, open(salida / "test_metrics.json", "w"), indent=2)
        if wandb_run:
            wandb_run.summary.update({f"test_{k}": v for k, v in metricas.items() if k != "matriz_confusion"})
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    main(parser.parse_args().config)
