"""
Training logic for the MangaInfiller.

Two-phase strategy:
  Phase 1: Train ControlNet adapter only (SD UNet frozen)
  Phase 2: Unfreeze SD UNet at low LR, fine-tune everything jointly

Uses Accelerate for mixed-precision and multi-GPU training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from mangaka.infiller.model import MangaInfiller, LayoutControlNet
from mangaka.infiller.dataset import MangaInfillerDataset

logger = logging.getLogger(__name__)


def _apply_lora_to_unet(unet: nn.Module, cfg: dict) -> nn.Module:
    """Wrap UNet with LoRA adapters on attention layers.

    Returns the PeftModel (unet is modified in-place).
    """
    from peft import LoraConfig, get_peft_model

    lora_cfg = cfg["lora"]
    config = LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"]),
    )
    peft_model = get_peft_model(unet, config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA applied: %d trainable params (%.2fM) / %d total (%.2fM)",
        trainable, trainable / 1e6, total, total / 1e6,
    )
    return peft_model


def train_infiller(
    image_dir: str,
    annotation_dir: str,
    cfg: dict,
    accelerator=None,
):
    """Full infiller training pipeline.

    Args:
        image_dir: path to manga images
        annotation_dir: path to annotation JSONs
        cfg: configuration dict (from infiller.yaml)
        accelerator: optional Accelerate accelerator (created if None)
    """
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    if accelerator is None:
        accelerator = Accelerator(
            mixed_precision="fp16" if cfg.get("amp", True) else "no",
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        )

    set_seed(cfg.get("seed", 42))
    device = accelerator.device

    # Dataset
    dataset = MangaInfillerDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        resolution=cfg.get("resolution", 512),
    )

    if len(dataset) == 0:
        logger.error("No samples found!")
        return

    val_split = cfg.get("val_split", 0.1)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )

    train_dl = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", 4),
        shuffle=True, num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.get("batch_size", 4),
        shuffle=False, num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    infiller = MangaInfiller(
        sd_model_id=cfg.get("sd_model", "stabilityai/stable-diffusion-2-inpainting"),
        resolution=cfg.get("resolution", 512),
    )
    infiller.load_sd_pipeline(device, dtype=torch.float32)  # fp32 for training

    controlnet = infiller.controlnet.to(device)
    unet = infiller.get_unet()
    vae = infiller.get_vae()
    text_encoder = infiller._text_encoder
    scheduler = infiller._scheduler

    # Freeze VAE and text encoder (never trained)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    save_dir = Path(cfg.get("save_path", "checkpoints/infiller"))
    save_dir.mkdir(parents=True, exist_ok=True)

    freeze_sd_epochs = cfg.get("freeze_sd_epochs", 20)
    total_epochs = cfg.get("epochs", 100)

    # Phase 1: ControlNet only
    if freeze_sd_epochs > 0:
        accelerator.print(f"=== Phase 1: ControlNet only ({freeze_sd_epochs} epochs) ===")
        unet.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            controlnet.parameters(),
            lr=cfg.get("controlnet_lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.01),
        )

        controlnet, optimizer, train_dl_acc = accelerator.prepare(
            controlnet, optimizer, train_dl,
        )

        _run_epochs(
            accelerator, unet, vae, text_encoder, scheduler,
            controlnet, optimizer, train_dl_acc, val_dl,
            freeze_sd_epochs, save_dir / "phase1", device,
        )

        accelerator.print("Phase 1 complete.")

    # Phase 2: Style adaptation (LoRA or full fine-tuning)
    remaining_epochs = total_epochs - freeze_sd_epochs
    lora_cfg = cfg.get("lora", {})
    use_lora = lora_cfg.get("enabled", False)

    if remaining_epochs > 0:
        if use_lora:
            accelerator.print(
                f"=== Phase 2: LoRA style adaptation ({remaining_epochs} epochs) ==="
            )
            unet.requires_grad_(False)
            unet = _apply_lora_to_unet(unet, cfg)

            lora_params = [p for p in unet.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([
                {"params": controlnet.parameters(), "lr": cfg.get("controlnet_lr", 1e-4)},
                {"params": lora_params, "lr": lora_cfg.get("lr", 1e-4)},
            ], weight_decay=cfg.get("weight_decay", 0.01))

            unet, controlnet, optimizer, train_dl_acc = accelerator.prepare(
                unet, controlnet, optimizer, train_dl,
            )

            _run_epochs(
                accelerator, unet, vae, text_encoder, scheduler,
                controlnet, optimizer, train_dl_acc, val_dl,
                remaining_epochs, save_dir / "phase2", device,
                save_lora=True,
            )
        else:
            accelerator.print(
                f"=== Phase 2: Joint fine-tuning ({remaining_epochs} epochs) ==="
            )
            unet.requires_grad_(True)

            optimizer = torch.optim.AdamW([
                {"params": controlnet.parameters(), "lr": cfg.get("controlnet_lr", 1e-4)},
                {"params": unet.parameters(), "lr": cfg.get("unfreeze_sd_lr", 1e-6)},
            ], weight_decay=cfg.get("weight_decay", 0.01))

            unet, controlnet, optimizer, train_dl_acc = accelerator.prepare(
                unet, controlnet, optimizer, train_dl,
            )

            _run_epochs(
                accelerator, unet, vae, text_encoder, scheduler,
                controlnet, optimizer, train_dl_acc, val_dl,
                remaining_epochs, save_dir / "phase2", device,
            )

    accelerator.print("Training complete.")


def _run_epochs(
    accelerator,
    unet, vae, text_encoder, scheduler,
    controlnet, optimizer, train_dl, val_dl,
    epochs, save_dir, device,
    save_lora: bool = False,
    lora_save_name: str = "unet_lora",
):
    """Inner training loop for one phase."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        # Train
        controlnet.train()
        unet.train()

        train_loss = 0.0
        train_n = 0

        for batch in train_dl:
            with accelerator.accumulate(controlnet):
                loss = _compute_loss(
                    batch, unet, vae, text_encoder, scheduler,
                    controlnet, device,
                )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item() * batch[0].size(0)
                train_n += batch[0].size(0)

        train_loss /= max(train_n, 1)

        # Validate
        controlnet.eval()
        unet.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = tuple(t.to(device) for t in batch)
                loss = _compute_loss(
                    batch, unet, vae, text_encoder, scheduler,
                    controlnet, device,
                )
                val_loss += loss.item() * batch[0].size(0)
                val_n += batch[0].size(0)

        val_loss /= max(val_n, 1)

        accelerator.print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            if accelerator.is_main_process:
                unwrapped_cn = accelerator.unwrap_model(controlnet)
                torch.save(unwrapped_cn.state_dict(), save_dir / "controlnet_best.pt")

                if save_lora:
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    lora_dir = save_dir / lora_save_name
                    unwrapped_unet.save_pretrained(str(lora_dir))
                    accelerator.print(
                        f"  -> saved best model + LoRA (val_loss={best_val:.4f})"
                    )
                else:
                    accelerator.print(
                        f"  -> saved best model (val_loss={best_val:.4f})"
                    )


def _compute_loss(batch, unet, vae, text_encoder, scheduler, controlnet, device):
    """Compute the denoising loss for one batch."""
    images, masks, layouts, input_ids = batch
    images = images.to(device)
    masks = masks.to(device)
    layouts = layouts.to(device)
    input_ids = input_ids.to(device)

    # Encode text
    with torch.no_grad():
        prompt_embeds = text_encoder(input_ids)[0]

    # Encode images to latents
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Sample noise and timesteps
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (latents.shape[0],), device=device,
    ).long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # Masked image latents
    latent_h, latent_w = latents.shape[2], latents.shape[3]
    mask_latent = F.interpolate(masks, size=(latent_h, latent_w), mode="nearest")

    masked_images = images * (1 - masks)
    with torch.no_grad():
        masked_latents = vae.encode(masked_images).latent_dist.sample()
        masked_latents = masked_latents * vae.config.scaling_factor

    # SD inpainting input
    latent_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)

    # ControlNet residuals (layout must be in latent resolution)
    layout_latent = F.interpolate(
        layouts, size=(latent_h, latent_w),
        mode="bilinear", align_corners=False,
    )
    down_residuals, mid_residual = controlnet(layout_latent)

    # Predict noise
    noise_pred = unet(
        latent_input,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        down_block_additional_residuals=down_residuals,
        mid_block_additional_residual=mid_residual,
    ).sample

    return F.mse_loss(noise_pred, noise)
