"""
Training logic for MangaDetectorNet — hierarchical detection + LM description.

Trains three heads in parallel:
  - Detection head: SmoothL1 bbox loss + CrossEntropy type loss
  - Description head: CrossEntropy language modelling loss

Two-phase training:
  Phase 1: Panel-only pretraining (learn coarse layout detection)
  Phase 2: Full hierarchical training (panels + elements)
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import save_model as safetensors_save

from mangaka.detector.model import (
    MangaDetectorNet,
    TYPE_NONE,
    NUM_TYPE_CLASSES,
    RETINA_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class DetectorLoss(nn.Module):
    """Joint detection + description loss.

    Detection:
      - bbox: SmoothL1 on non-NONE samples (normalised to [0,1])
      - type: CrossEntropy with optional label smoothing

    Description:
      - The LM head computes its own cross-entropy loss internally.
    """

    def __init__(
        self,
        bbox_weight: float = 1.0,
        type_weight: float = 1.0,
        desc_weight: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.type_weight = type_weight
        self.desc_weight = desc_weight
        self.type_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bbox_loss_fn = nn.SmoothL1Loss()

    def forward(
        self,
        bbox_pred: torch.Tensor,
        type_pred: torch.Tensor,
        desc_loss: torch.Tensor | None,
        target: torch.Tensor,
    ):
        """
        Args:
            bbox_pred: (B, 4) predicted bbox in [0, 1]
            type_pred: (B, NUM_TYPE_CLASSES) type logits
            desc_loss: scalar LM language modelling loss, or None
            target:    (B, 5) ground truth (x1, y1, x2, y2, type_id) in retina coords

        Returns:
            total, bbox_loss, type_loss, desc_loss (all scalars)
        """
        target_type = target[:, 4].long()

        # Type classification loss (all samples)
        type_loss = self.type_loss_fn(type_pred, target_type)

        # Bbox loss (non-NONE samples only)
        non_none = target_type != TYPE_NONE
        if non_none.any():
            pred_norm = bbox_pred[non_none]
            tgt_norm = target[non_none, :4] / RETINA_SIZE
            bbox_loss = self.bbox_loss_fn(pred_norm, tgt_norm)
        else:
            bbox_loss = torch.tensor(0.0, device=bbox_pred.device)

        # Description loss
        if desc_loss is None:
            desc_loss = torch.tensor(0.0, device=bbox_pred.device)

        total = (
            self.bbox_weight * bbox_loss
            + self.type_weight * type_loss
            + self.desc_weight * desc_loss
        )
        return total, bbox_loss, type_loss, desc_loss


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model: nn.Module,
    loss_fn: DetectorLoss,
    batch: tuple,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    clip_grad: float = 0.0,
    device: torch.device = torch.device("cpu"),
):
    """Run one forward/backward pass."""
    img, prev, target, desc_ids, desc_mask, target_retina = [
        t.to(device) for t in batch
    ]

    use_amp = scaler is not None

    with torch.amp.autocast("cuda", enabled=use_amp):
        bbox_pred, type_pred, desc_loss = model(
            img, prev,
            desc_input_ids=desc_ids,
            desc_attention_mask=desc_mask,
            target_bbox_for_roi=target_retina,
        )
        total, bbox_l, type_l, desc_l = loss_fn(
            bbox_pred, type_pred, desc_loss, target,
        )

    if optimizer is not None:
        if use_amp:
            scaler.scale(total).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        optimizer.zero_grad()

    B = img.size(0)
    return (
        total.item(), bbox_l.item(), type_l.item(),
        desc_l.item() if isinstance(desc_l, torch.Tensor) else desc_l,
        B,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def fit(
    epochs: int,
    model: nn.Module,
    loss_fn: DetectorLoss,
    optimizer: torch.optim.Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: torch.device,
    save_path: str,
    patience: int = 15,
    scaler: torch.amp.GradScaler | None = None,
    clip_grad: float = 0.0,
    warmup_epochs: int = 0,
):
    """Main training loop with cosine annealing + optional warmup."""
    best_val = float("inf")
    no_improve = 0
    use_ddp = dist.is_initialized()

    # LR scheduler
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1),
        )

    for epoch in range(epochs):
        if use_ddp and hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(epoch)

        # --- Train ---
        model.train()
        t_tot, t_bb, t_tp, t_desc, t_n = 0.0, 0.0, 0.0, 0.0, 0
        for batch in train_dl:
            tot, bb, tp, desc, n = train_step(
                model, loss_fn, batch, optimizer, scaler, clip_grad, device,
            )
            t_tot += tot * n
            t_bb += bb * n
            t_tp += tp * n
            t_desc += desc * n
            t_n += n
        t_tot /= t_n
        t_bb /= t_n
        t_tp /= t_n
        t_desc /= t_n

        # --- Validate ---
        model.eval()
        v_tot, v_bb, v_tp, v_desc, v_n = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                tot, bb, tp, desc, n = train_step(
                    model, loss_fn, batch, device=device,
                )
                v_tot += tot * n
                v_bb += bb * n
                v_tp += tp * n
                v_desc += desc * n
                v_n += n
        v_tot /= v_n
        v_bb /= v_n
        v_tp /= v_n
        v_desc /= v_n

        # Average across ranks
        if use_ddp:
            val_t = torch.tensor([v_tot, v_bb, v_tp, v_desc], device=device)
            dist.all_reduce(val_t, op=dist.ReduceOp.AVG)
            v_tot, v_bb, v_tp, v_desc = val_t.tolist()

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if is_main():
            logger.info(
                "Epoch %3d | train %.4f (bbox=%.4f type=%.4f desc=%.4f) | "
                "val %.4f (bbox=%.4f type=%.4f desc=%.4f) | lr=%.2e",
                epoch, t_tot, t_bb, t_tp, t_desc,
                v_tot, v_bb, v_tp, v_desc, lr,
            )

        # Checkpoint best
        if v_tot < best_val:
            best_val = v_tot
            no_improve = 0
            if is_main():
                raw = model.module if use_ddp else model
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                safetensors_save(raw, save_path)
                logger.info("  -> saved best model (%s)", save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                if is_main():
                    logger.info(
                        "  Early stopping (no improvement for %d epochs, best=%.4f)",
                        patience, best_val,
                    )
                break

        if use_ddp:
            dist.barrier()


# ---------------------------------------------------------------------------
# Build optimizer with per-group LRs
# ---------------------------------------------------------------------------

def build_optimizer(model: MangaDetectorNet, lr: float, backbone_lr_scale: float = 0.1,
                    weight_decay: float = 0.01):
    """Create AdamW optimizer with separate LRs for backbone / heads."""
    backbone_params = (
        list(model.stem.parameters())
        + list(model.layer1.parameters())
        + list(model.layer2.parameters())
        + list(model.layer3.parameters())
        + list(model.layer4.parameters())
    )
    detection_params = (
        list(model.fc_shared.parameters())
        + list(model.bbox_head.parameters())
        + list(model.type_head.parameters())
        + list(model.roi_proj.parameters())
    )
    description_params = list(model.description_head.parameters())

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * backbone_lr_scale},
        {"params": detection_params, "lr": lr},
        {"params": description_params, "lr": lr},
    ], weight_decay=weight_decay)


def build_dataloaders(
    dataset,
    val_split: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
):
    """Split dataset and build train/val dataloaders (DDP-aware)."""
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    use_ddp = dist.is_initialized()
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dl = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_dl, val_dl
