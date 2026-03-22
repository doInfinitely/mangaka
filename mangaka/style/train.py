"""Training loop for the style transfer GAN.

Uses Accelerate for mixed-precision and multi-GPU training.
Alternates between discriminator and generator updates each step.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from mangaka.style.model import StyleTransferGenerator
from mangaka.style.discriminator import PatchDiscriminator
from mangaka.style.losses import CombinedStyleLoss
from mangaka.style.dataset import StyleTransferDataset, collate_style_batch

logger = logging.getLogger(__name__)


def train_style_gan(
    image_dir: str,
    bank_dir: str,
    cfg: dict,
    accelerator=None,
):
    """Full style transfer GAN training pipeline.

    Args:
        image_dir: path to artist image directories
        bank_dir: path to precomputed style bank .pt files
        cfg: configuration dict (from style.yaml)
        accelerator: optional Accelerate accelerator
    """
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    if accelerator is None:
        accelerator = Accelerator(
            mixed_precision="fp16" if cfg.get("amp", True) else "no",
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        )

    set_seed(cfg.get("seed", 42))

    # Dataset
    dataset = StyleTransferDataset(
        image_dir=image_dir,
        bank_dir=bank_dir,
        resolution=cfg.get("resolution", 512),
        max_bank_size=cfg.get("max_bank_size", 256),
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
        train_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_style_batch,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_style_batch,
    )

    # Models
    gen_cfg = cfg.get("generator", {})
    generator = StyleTransferGenerator(
        base_channels=gen_cfg.get("base_channels", 64),
        style_dim=cfg.get("style_dim", 512),
        num_heads=gen_cfg.get("num_heads", 8),
    )

    disc_cfg = cfg.get("discriminator", {})
    discriminator = PatchDiscriminator(
        base_channels=disc_cfg.get("base_channels", 64),
        n_layers=disc_cfg.get("n_layers", 3),
    )

    # Losses
    loss_cfg = cfg.get("losses", {})
    criterion = CombinedStyleLoss(
        lambda_l1=loss_cfg.get("lambda_l1", 10.0),
        lambda_perceptual=loss_cfg.get("lambda_perceptual", 1.0),
        lambda_style=loss_cfg.get("lambda_style", 50.0),
        lambda_adversarial=loss_cfg.get("lambda_adversarial", 1.0),
    )

    # Optimizers
    opt_g = torch.optim.AdamW(
        generator.parameters(),
        lr=cfg.get("generator_lr", 1e-4),
        betas=(0.5, 0.999),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=cfg.get("discriminator_lr", 4e-4),
        betas=(0.5, 0.999),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    # Prepare with Accelerate
    generator, discriminator, opt_g, opt_d, train_dl = accelerator.prepare(
        generator, discriminator, opt_g, opt_d, train_dl,
    )

    # Move VGG losses to device (frozen, not in optimizer)
    criterion.perceptual.to(accelerator.device)
    criterion.style.to(accelerator.device)

    save_dir = Path(cfg.get("save_path", "checkpoints/style"))
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg.get("epochs", 100)
    best_val_g = float("inf")

    # Training log (written to save_dir for persistent metrics)
    log_path = save_dir / "training_log.jsonl"

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_g_l1 = 0.0
        epoch_g_perc = 0.0
        epoch_g_style = 0.0
        epoch_g_adv = 0.0
        epoch_n = 0
        nan_steps = 0

        for batch in train_dl:
            content = batch["content"]
            target = batch["target"]
            style_bank = batch["style_bank"]
            bs = content.shape[0]

            # ----- Discriminator step -----
            opt_d.zero_grad()
            with accelerator.accumulate(discriminator):
                with torch.no_grad():
                    fake = generator(content, style_bank)

                disc_real = discriminator(target)
                disc_fake = discriminator(fake.detach())
                d_loss = criterion.discriminator_loss(disc_real, disc_fake)

                accelerator.backward(d_loss)
                opt_d.step()

            # ----- Generator step -----
            opt_g.zero_grad()
            opt_d.zero_grad()  # clear D grads that leak from G backward
            with accelerator.accumulate(generator):
                fake = generator(content, style_bank)
                disc_fake = discriminator(fake)

                g_losses = criterion.generator_loss(
                    pred=fake, target=target, disc_fake=disc_fake,
                )
                g_loss = g_losses["total"]

                if torch.isnan(g_loss) or torch.isinf(g_loss):
                    nan_steps += 1
                    continue

                accelerator.backward(g_loss)
                opt_g.step()

            epoch_d_loss += d_loss.item() * bs
            epoch_g_loss += g_loss.item() * bs
            epoch_g_l1 += g_losses["l1"].item() * bs
            epoch_g_perc += g_losses["perceptual"].item() * bs
            epoch_g_style += g_losses["style"].item() * bs
            epoch_g_adv += g_losses["adversarial"].item() * bs
            epoch_n += bs

        epoch_d_loss /= max(epoch_n, 1)
        epoch_g_loss /= max(epoch_n, 1)
        epoch_g_l1 /= max(epoch_n, 1)
        epoch_g_perc /= max(epoch_n, 1)
        epoch_g_style /= max(epoch_n, 1)
        epoch_g_adv /= max(epoch_n, 1)

        # Validation
        generator.eval()
        discriminator.eval()
        val_g_loss = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_dl:
                content = batch["content"].to(accelerator.device)
                target = batch["target"].to(accelerator.device)
                style_bank = batch["style_bank"].to(accelerator.device)
                bs = content.shape[0]

                fake = generator(content, style_bank)
                disc_fake = discriminator(fake)
                g_losses = criterion.generator_loss(
                    pred=fake, target=target, disc_fake=disc_fake,
                )
                val_g_loss += g_losses["total"].item() * bs
                val_n += bs

        val_g_loss /= max(val_n, 1)

        # Check GradScaler state if using AMP
        scaler_scale = None
        if hasattr(accelerator, "scaler") and accelerator.scaler is not None:
            scaler_scale = accelerator.scaler.get_scale()

        accelerator.print(
            f"Epoch {epoch:3d} | D={epoch_d_loss:.4f} | G={epoch_g_loss:.4f} "
            f"(l1={epoch_g_l1:.4f} perc={epoch_g_perc:.4f} "
            f"style={epoch_g_style:.4f} adv={epoch_g_adv:.4f}) | "
            f"val_G={val_g_loss:.4f} | nan_steps={nan_steps}"
            + (f" | scale={scaler_scale:.0f}" if scaler_scale is not None else "")
        )

        # Append to log file
        if accelerator.is_main_process:
            import json
            log_entry = {
                "epoch": epoch,
                "d_loss": epoch_d_loss,
                "g_loss": epoch_g_loss,
                "g_l1": epoch_g_l1,
                "g_perceptual": epoch_g_perc,
                "g_style": epoch_g_style,
                "g_adversarial": epoch_g_adv,
                "val_g_loss": val_g_loss,
                "nan_steps": nan_steps,
                "scaler_scale": scaler_scale,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # Save best
        if val_g_loss < best_val_g:
            best_val_g = val_g_loss
            if accelerator.is_main_process:
                g_unwrapped = accelerator.unwrap_model(generator)
                d_unwrapped = accelerator.unwrap_model(discriminator)
                torch.save(g_unwrapped.state_dict(), save_dir / "generator_best.pt")
                torch.save(d_unwrapped.state_dict(), save_dir / "discriminator_best.pt")
                accelerator.print(f"  -> saved best (val_G={best_val_g:.4f})")

    accelerator.print("Style GAN training complete.")
