"""
CLI: Train the MangaDetectorNet.

Usage:
    torchrun --nproc_per_node=4 scripts/train_detector.py \
        --image-dir /path/to/manga \
        --annotation-dir /path/to/annotations \
        --config configs/detector.yaml

Single GPU:
    python scripts/train_detector.py \
        --image-dir /path/to/manga \
        --annotation-dir /path/to/annotations
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mangaka.train_detector")


def main():
    parser = argparse.ArgumentParser(description="Train MangaDetectorNet")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--config", default="configs/detector.yaml")

    # Override config values from CLI
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides
    for key in ("epochs", "pretrain_epochs", "batch_size", "lr", "backbone", "save_path", "seed"):
        cli_val = getattr(args, key.replace("-", "_"), None)
        if cli_val is not None:
            cfg[key] = cli_val

    # Defaults
    cfg.setdefault("backbone", "resnet50")
    cfg.setdefault("lm_model", "Qwen/Qwen2.5-0.5B")
    cfg.setdefault("prefix_length", 8)
    cfg.setdefault("max_description_length", 64)
    cfg.setdefault("epochs", 50)
    cfg.setdefault("pretrain_epochs", 10)
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("lr", 2e-4)
    cfg.setdefault("backbone_lr_scale", 0.1)
    cfg.setdefault("weight_decay", 0.01)
    cfg.setdefault("warmup_epochs", 5)
    cfg.setdefault("patience", 15)
    cfg.setdefault("bbox_weight", 1.0)
    cfg.setdefault("type_weight", 1.0)
    cfg.setdefault("description_weight", 0.5)
    cfg.setdefault("val_split", 0.15)
    cfg.setdefault("amp", True)
    cfg.setdefault("clip_grad", 1.0)
    cfg.setdefault("label_smoothing", 0.1)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("seed", 42)
    cfg.setdefault("save_path", "checkpoints/detector.safetensors")
    cfg.setdefault("freeze_lm_epochs", 5)

    # Seed
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # DDP setup
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")

    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        from mangaka.detector.train import setup_ddp, is_main
        local_rank = setup_ddp()
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    from mangaka.detector.train import (
        cleanup_ddp, is_main, DetectorLoss, fit,
        build_optimizer, build_dataloaders,
    )
    from mangaka.detector.model import MangaDetectorNet
    from mangaka.detector.dataset import MangaDetectorDataset, PanelOnlyDataset

    if is_main():
        logger.info("Config: %s", cfg)
        logger.info("Device: %s", device)

    # Model
    model = MangaDetectorNet(
        backbone=cfg["backbone"],
        lm_model=cfg["lm_model"],
        prefix_length=cfg["prefix_length"],
        max_description_length=cfg["max_description_length"],
    ).to(device)

    if use_ddp:
        import torch.nn as nn
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True,
        )

    if is_main():
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Model params: %.1fM", n_params / 1e6)

    scaler = torch.amp.GradScaler("cuda") if cfg["amp"] else None

    # Dataset
    full_dataset = MangaDetectorDataset(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        max_desc_len=cfg["max_description_length"],
    )

    if len(full_dataset) == 0:
        logger.error("No samples found! Check --image-dir and --annotation-dir.")
        sys.exit(1)

    # Loss
    loss_fn = DetectorLoss(
        bbox_weight=cfg["bbox_weight"],
        type_weight=cfg["type_weight"],
        desc_weight=cfg["description_weight"],
        label_smoothing=cfg["label_smoothing"],
    ).to(device)

    raw_model = model.module if use_ddp else model

    # --- Phase 1: Panel pretraining ---
    if cfg["pretrain_epochs"] > 0:
        if is_main():
            logger.info("=== Phase 1: Panel pretraining (%d epochs) ===", cfg["pretrain_epochs"])

        panel_ds = PanelOnlyDataset(full_dataset)
        panel_train_dl, panel_val_dl = build_dataloaders(
            panel_ds, cfg["val_split"], cfg["batch_size"],
            cfg["num_workers"], cfg["seed"],
        )

        optimizer = build_optimizer(
            raw_model, cfg["lr"], cfg["backbone_lr_scale"], cfg["weight_decay"],
        )

        fit(
            cfg["pretrain_epochs"], model, loss_fn, optimizer,
            panel_train_dl, panel_val_dl, device, cfg["save_path"],
            patience=cfg["pretrain_epochs"],
            scaler=scaler, clip_grad=cfg["clip_grad"],
            warmup_epochs=cfg["warmup_epochs"],
        )

        if is_main():
            logger.info("Panel pretraining complete.")

    # --- Phase 2: Full hierarchical training ---
    if is_main():
        logger.info("=== Phase 2: Full hierarchical training (%d epochs) ===", cfg["epochs"])

    train_dl, val_dl = build_dataloaders(
        full_dataset, cfg["val_split"], cfg["batch_size"],
        cfg["num_workers"], cfg["seed"] + 1,
    )

    optimizer = build_optimizer(
        raw_model, cfg["lr"], cfg["backbone_lr_scale"], cfg["weight_decay"],
    )

    fit(
        cfg["epochs"], model, loss_fn, optimizer,
        train_dl, val_dl, device, cfg["save_path"],
        patience=cfg["patience"],
        scaler=scaler, clip_grad=cfg["clip_grad"],
        warmup_epochs=cfg["warmup_epochs"],
    )

    if use_ddp:
        cleanup_ddp()

    if is_main():
        logger.info("Done. Model saved to %s", cfg["save_path"])


if __name__ == "__main__":
    main()
