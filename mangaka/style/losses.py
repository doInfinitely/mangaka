"""Loss functions for style transfer GAN training.

Combines:
    - L1 reconstruction loss
    - Perceptual loss (VGG feature matching)
    - Style loss (Gram matrix matching)
    - Adversarial loss (hinge GAN)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mangaka.style.encoder import VGGFeatureExtractor


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for style loss.

    Args:
        x: (B, C, H, W) feature map

    Returns:
        (B, C, C) Gram matrix (normalised by spatial size)
    """
    B, C, H, W = x.shape
    feats = x.view(B, C, H * W)
    gram = torch.bmm(feats, feats.transpose(1, 2))
    return gram / (C * H * W)


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss — L1 distance between feature maps."""

    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 3, H, W) generated image in [0, 1]
            target: (B, 3, H, W) ground truth image in [0, 1]

        Returns:
            Scalar perceptual loss
        """
        # Force fp32 — VGG features can overflow fp16
        with torch.amp.autocast("cuda", enabled=False):
            pred = pred.float()
            target = target.float()
            pred_feats = self.vgg(pred)
            target_feats = self.vgg(target)

            loss = torch.tensor(0.0, device=pred.device)
            for pf, tf in zip(pred_feats, target_feats):
                loss = loss + F.l1_loss(pf, tf)
            return loss


class StyleLoss(nn.Module):
    """Gram-matrix style loss across VGG feature layers."""

    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 3, H, W) generated image in [0, 1]
            target: (B, 3, H, W) style reference image in [0, 1]

        Returns:
            Scalar style loss
        """
        # Force fp32 — Gram matrix bmm overflows fp16
        with torch.amp.autocast("cuda", enabled=False):
            pred = pred.float()
            target = target.float()
            pred_feats = self.vgg(pred)
            target_feats = self.vgg(target)

            loss = torch.tensor(0.0, device=pred.device)
            for pf, tf in zip(pred_feats, target_feats):
                loss = loss + F.l1_loss(gram_matrix(pf), gram_matrix(tf))
            return loss


class CombinedStyleLoss(nn.Module):
    """Combined loss for style transfer GAN training.

    Total = λ_l1 * L1 + λ_perc * perceptual + λ_style * style + λ_adv * adversarial
    """

    def __init__(
        self,
        lambda_l1: float = 10.0,
        lambda_perceptual: float = 1.0,
        lambda_style: float = 50.0,
        lambda_adversarial: float = 1.0,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style
        self.lambda_adversarial = lambda_adversarial

        self.perceptual = PerceptualLoss()
        self.style = StyleLoss()

    def generator_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        disc_fake: torch.Tensor,
        style_ref: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all generator losses.

        Args:
            pred: (B, 3, H, W) generated image in [-1, 1]
            target: (B, 3, H, W) ground truth image in [-1, 1]
            disc_fake: discriminator output on pred
            style_ref: (B, 3, H, W) style reference in [-1, 1] (defaults to target)

        Returns:
            Dict with 'total', 'l1', 'perceptual', 'style', 'adversarial' losses
        """
        # Convert [-1,1] → [0,1] for VGG
        pred_01 = (pred + 1) / 2
        target_01 = (target + 1) / 2
        style_ref_01 = (style_ref + 1) / 2 if style_ref is not None else target_01

        l1 = F.l1_loss(pred, target)
        perc = self.perceptual(pred_01, target_01)
        style = self.style(pred_01, style_ref_01)
        adv = hinge_generator_loss(disc_fake)

        total = (
            self.lambda_l1 * l1
            + self.lambda_perceptual * perc
            + self.lambda_style * style
            + self.lambda_adversarial * adv
        )

        return {
            "total": total,
            "l1": l1,
            "perceptual": perc,
            "style": style,
            "adversarial": adv,
        }

    @staticmethod
    def discriminator_loss(
        disc_real: torch.Tensor,
        disc_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Hinge loss for discriminator."""
        return hinge_discriminator_loss(disc_real, disc_fake)


def hinge_generator_loss(disc_fake: torch.Tensor) -> torch.Tensor:
    """Hinge GAN generator loss: -E[D(G(x))]."""
    return -disc_fake.mean()


def hinge_discriminator_loss(
    disc_real: torch.Tensor,
    disc_fake: torch.Tensor,
) -> torch.Tensor:
    """Hinge GAN discriminator loss."""
    real_loss = F.relu(1.0 - disc_real).mean()
    fake_loss = F.relu(1.0 + disc_fake).mean()
    return (real_loss + fake_loss) / 2
