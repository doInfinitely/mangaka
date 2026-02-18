"""
SD-Conditioned Infiller — Stable Diffusion inpainting with ControlNet layout.

Hierarchical rendering descends the containment hierarchy, giving each level
the full SD resolution — just like nissl's retina-based infiller:

  Page level:    page at full res → infill each panel region
  Panel level:   crop panel at full res → infill each element region
  Element level: crop element at full res → fine-detail rendering

At every level the model sees the full pixel budget for the region it is
rendering, producing high-fidelity output at all scales.

The ControlNet adapter encodes spatial layout (borders, fills, element types)
at the current hierarchy level as additional conditioning for the SD UNet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from mangaka.schema import MangaPage, Panel, MangaElement
from mangaka.infiller.layout import (
    render_page_layout,
    render_panel_layout,
    render_element_layout,
    render_panel_mask,
    render_element_mask_in_panel,
    render_full_mask,
    NUM_LAYOUT_CHANNELS,
)
from mangaka.utils.bbox import denormalise_bbox

logger = logging.getLogger(__name__)


class LayoutControlNet(nn.Module):
    """Lightweight ControlNet-style adapter for layout conditioning.

    Takes a multi-channel layout map in **latent space** and produces 12
    residual tensors matching the SD v1 UNet's down_block_res_samples:

        [ 0] 320 @ 64x64  (conv_in)
        [ 1] 320 @ 64x64  (down_block_0 resnet 0)
        [ 2] 320 @ 64x64  (down_block_0 resnet 1)
        [ 3] 320 @ 32x32  (down_block_0 downsample)
        [ 4] 640 @ 32x32  (down_block_1 resnet 0)
        [ 5] 640 @ 32x32  (down_block_1 resnet 1)
        [ 6] 640 @ 16x16  (down_block_1 downsample)
        [ 7] 1280 @ 16x16 (down_block_2 resnet 0)
        [ 8] 1280 @ 16x16 (down_block_2 resnet 1)
        [ 9] 1280 @ 8x8   (down_block_2 downsample)
        [10] 1280 @ 8x8   (down_block_3 resnet 0)
        [11] 1280 @ 8x8   (down_block_3 resnet 1)

    The caller must resize the layout to latent resolution before passing it
    (i.e. resolution // 8, typically 64x64 for 512px images).
    """

    def __init__(self, in_channels: int = NUM_LAYOUT_CHANNELS, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels

        # Encoder at latent resolution (64x64 for 512px images)
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        # Downsampling: 64→32→16→8
        self.down1 = self._make_down_block(base_channels, base_channels * 2)
        self.down2 = self._make_down_block(base_channels * 2, base_channels * 4)
        self.down3 = self._make_down_block(base_channels * 4, base_channels * 8)

        # Zero-init 1x1 convolutions — one per UNet residual position (12 down + 1 mid)
        self.zc = nn.ModuleList([
            self._zero_conv(base_channels, 320),       # [0]  320@64
            self._zero_conv(base_channels, 320),       # [1]  320@64
            self._zero_conv(base_channels, 320),       # [2]  320@64
            self._zero_conv(base_channels * 2, 320),   # [3]  320@32
            self._zero_conv(base_channels * 2, 640),   # [4]  640@32
            self._zero_conv(base_channels * 2, 640),   # [5]  640@32
            self._zero_conv(base_channels * 4, 640),   # [6]  640@16
            self._zero_conv(base_channels * 4, 1280),  # [7]  1280@16
            self._zero_conv(base_channels * 4, 1280),  # [8]  1280@16
            self._zero_conv(base_channels * 8, 1280),  # [9]  1280@8
            self._zero_conv(base_channels * 8, 1280),  # [10] 1280@8
            self._zero_conv(base_channels * 8, 1280),  # [11] 1280@8
        ])
        # Mid block residual (1280 @ 8x8)
        self.mid_zc = self._zero_conv(base_channels * 8, 1280)

    @staticmethod
    def _make_down_block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
        )

    @staticmethod
    def _zero_conv(in_ch: int, out_ch: int) -> nn.Module:
        conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv

    def forward(self, layout: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Produce 12 down-block + 1 mid-block residuals for the SD v1 UNet.

        Args:
            layout: (B, C, H, W) layout in latent resolution (e.g. 64x64)

        Returns:
            (down_residuals, mid_residual): 12 down-block tensors + 1 mid-block tensor
        """
        h64 = self.input_conv(layout)  # (B, 64, H, W)
        h32 = self.down1(h64)          # (B, 128, H/2, W/2)
        h16 = self.down2(h32)          # (B, 256, H/4, W/4)
        h8 = self.down3(h16)           # (B, 512, H/8, W/8)

        feats = [h64, h64, h64, h32, h32, h32, h16, h16, h16, h8, h8, h8]
        down = [self.zc[i](f) for i, f in enumerate(feats)]
        mid = self.mid_zc(h8)
        return down, mid


DEFAULT_STYLE_PREFIX = "manga, black and white, ink drawing, screentone shading, clean linework"
DEFAULT_NEGATIVE_PROMPT = "photorealistic, 3d render, color, photograph, blurry, low quality, watermark"
DEFAULT_SEED = 42


class MangaInfiller(nn.Module):
    """Manga infiller using SD inpainting + ControlNet layout conditioning.

    Wraps the diffusers StableDiffusionInpaintPipeline with an additional
    ControlNet-style layout adapter.

    Rendering descends the containment hierarchy:

      1. Page level — render the full page with panel-level layout conditioning.
         Each panel region is inpainted at full SD resolution using the page
         canvas as context and the panel description as the text prompt.

      2. Panel level — crop each rendered panel to full SD resolution.
         Within the panel, each element's bbox region is inpainted using the
         panel-level layout (element borders + type masks) as conditioning
         and the element description as the text prompt.

      3. Element level (optional) — crop each element to full SD resolution
         for fine-detail rendering.  Useful for characters and complex SFX.

    At every level the target region gets the full pixel budget, producing
    high-quality output at all scales of the hierarchy.
    """

    def __init__(
        self,
        sd_model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        resolution: int = 512,
        controlnet: LayoutControlNet | None = None,
        element_level_types: tuple[str, ...] = ("character", "sfx"),
    ):
        super().__init__()
        self.sd_model_id = sd_model_id
        self.resolution = resolution
        self.element_level_types = element_level_types

        # ControlNet layout adapter
        self.controlnet = controlnet or LayoutControlNet()

        # SD pipeline components loaded lazily
        self._pipe = None
        self._unet = None
        self._vae = None
        self._text_encoder = None
        self._tokenizer = None
        self._scheduler = None

    def load_sd_pipeline(self, device: torch.device, dtype: torch.dtype = torch.float16):
        """Load the Stable Diffusion inpainting pipeline."""
        from diffusers import StableDiffusionInpaintPipeline

        logger.info("Loading SD inpainting pipeline: %s", self.sd_model_id)
        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.sd_model_id,
            torch_dtype=dtype,
        ).to(device)

        self._unet = self._pipe.unet
        self._vae = self._pipe.vae
        self._text_encoder = self._pipe.text_encoder
        self._tokenizer = self._pipe.tokenizer
        self._scheduler = self._pipe.scheduler
        logger.info("SD pipeline loaded on %s", device)

    def load_lora_weights(self, lora_path: str | Path):
        """Load LoRA adapter weights onto the UNet.

        Must be called after load_sd_pipeline().
        """
        from peft import PeftModel

        lora_path = Path(lora_path)
        if self._pipe is None:
            raise RuntimeError(
                "SD pipeline not loaded. Call load_sd_pipeline() before load_lora_weights()."
            )
        logger.info("Loading LoRA weights from %s", lora_path)
        self._unet = PeftModel.from_pretrained(self._unet, str(lora_path))
        self._pipe.unet = self._unet
        logger.info("LoRA weights loaded")

    @property
    def pipe(self):
        if self._pipe is None:
            raise RuntimeError(
                "SD pipeline not loaded. Call load_sd_pipeline() first."
            )
        return self._pipe

    def get_unet(self) -> nn.Module:
        return self.pipe.unet

    def get_vae(self) -> nn.Module:
        return self.pipe.vae

    # -- Training forward: compute diffusion loss ----------------------------

    def training_step(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        layout_maps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the denoising training loss.

        Training samples are already cropped to the appropriate hierarchy
        level by the dataset, so this just runs the standard SD inpainting
        loss with ControlNet conditioning.

        Args:
            images: (B, 3, H, W) target images in [-1, 1]
            masks: (B, 1, H, W) inpainting masks (1 = inpaint region)
            layout_maps: (B, NUM_LAYOUT_CHANNELS, H, W) layout conditioning
            prompt_embeds: (B, seq_len, hidden_dim) text embeddings
            noise: optional pre-generated noise
            timesteps: optional specific timesteps

        Returns:
            Scalar MSE denoising loss
        """
        import torch.nn.functional as F

        unet = self._unet
        vae = self._vae
        scheduler = self._scheduler

        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        if noise is None:
            noise = torch.randn_like(latents)
        if timesteps is None:
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=latents.device,
            ).long()

        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        latent_h, latent_w = latents.shape[2], latents.shape[3]
        mask_latent = F.interpolate(masks, size=(latent_h, latent_w), mode="nearest")

        masked_images = images * (1 - masks)
        masked_latents = vae.encode(masked_images).latent_dist.sample()
        masked_latents = masked_latents * vae.config.scaling_factor

        latent_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)

        layout_latent = F.interpolate(
            layout_maps, size=(latent_h, latent_w),
            mode="bilinear", align_corners=False,
        )
        down_residuals, mid_residual = self.controlnet(layout_latent)

        noise_pred = unet(
            latent_input, timesteps,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
        ).sample

        return F.mse_loss(noise_pred, noise)

    # =====================================================================
    # Inference: hierarchical rendering descending the containment tree
    # =====================================================================

    @torch.no_grad()
    def render_page(
        self,
        page: MangaPage,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        style_override: str | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Render a full manga page from JSON by descending the hierarchy.

        Level 1 — Page:  white canvas → infill each panel at full resolution
        Level 2 — Panel: crop panel → infill each element at full resolution
        Level 3 — Element (optional): crop element → fine-detail infill

        Args:
            page: structured manga page representation
            num_inference_steps: SD denoising steps
            guidance_scale: classifier-free guidance scale
            style_override: if set, overrides page.style for the style prefix
            negative_prompt: if set, overrides the default negative prompt
            seed: if set, overrides the default seed for reproducibility

        Returns:
            PIL Image of the rendered page at (page.width, page.height)
        """
        res = self.resolution
        w, h = page.width, page.height

        # Resolve style, negative prompt, and seed
        style_prefix = style_override or page.style or DEFAULT_STYLE_PREFIX
        neg_prompt = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
        seed = seed if seed is not None else DEFAULT_SEED
        device = self.pipe.device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Start with a white canvas at page dimensions
        canvas = Image.new("RGB", (w, h), (255, 255, 255))

        # ---------------------------------------------------------------
        # Level 1: render each panel into the page canvas
        # ---------------------------------------------------------------
        for pi, panel in enumerate(page.panels):
            px1, py1, px2, py2 = denormalise_bbox(panel.bbox, w, h)
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)
            pw, ph = px2 - px1, py2 - py1
            if pw <= 0 or ph <= 0:
                continue

            # Crop the panel region from the current canvas (provides context
            # from previously rendered panels and panel borders)
            panel_context = canvas.crop((px1, py1, px2, py2))
            panel_context_res = panel_context.resize((res, res), Image.LANCZOS)

            # Full-canvas mask: inpaint the entire panel region
            mask_np = render_panel_mask(panel, res)
            mask_pil = Image.fromarray((mask_np[0] * 255).astype(np.uint8), mode="L")

            # Page-level layout (panel borders + fills)
            # Cropped to this panel's view of the page
            page_layout_np = render_page_layout(page, res)

            rendered_panel = self._infill(
                context=panel_context_res,
                mask=mask_pil,
                layout=page_layout_np,
                prompt=panel.description,
                steps=num_inference_steps,
                guidance=guidance_scale,
                style_prefix=style_prefix,
                negative_prompt=neg_prompt,
                generator=generator,
            )

            # ---------------------------------------------------------------
            # Level 2: render each element within the panel
            # ---------------------------------------------------------------
            rendered_panel = self._render_elements_in_panel(
                panel_image=rendered_panel,
                panel=panel,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                style_prefix=style_prefix,
                negative_prompt=neg_prompt,
                generator=generator,
            )

            # Paste the rendered panel back into the page canvas
            rendered_panel_sized = rendered_panel.resize((pw, ph), Image.LANCZOS)
            canvas.paste(rendered_panel_sized, (px1, py1))

        return canvas

    def _render_elements_in_panel(
        self,
        panel_image: Image.Image,
        panel: Panel,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        style_prefix: str = DEFAULT_STYLE_PREFIX,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        generator: torch.Generator | None = None,
    ) -> Image.Image:
        """Level 2: render all elements within a panel crop.

        *panel_image* is the panel rendered at self.resolution.
        For each element, we mask its bbox region and inpaint conditioned
        on the element description + panel-level layout.

        Elements whose type is in self.element_level_types also get a
        Level 3 fine-detail pass.
        """
        res = self.resolution
        current = panel_image.copy()

        # Panel-level layout showing element borders and types
        panel_layout_np = render_panel_layout(panel, res)

        for ei, elem in enumerate(panel.elements):
            # Mask for this element within the panel
            mask_np = render_element_mask_in_panel(panel, ei, res)
            mask_pil = Image.fromarray((mask_np[0] * 255).astype(np.uint8), mode="L")

            prompt = elem.description
            if elem.text:
                prompt += f' with text "{elem.text}"'

            rendered = self._infill(
                context=current,
                mask=mask_pil,
                layout=panel_layout_np,
                prompt=prompt,
                steps=num_inference_steps,
                guidance=guidance_scale,
                style_prefix=style_prefix,
                negative_prompt=negative_prompt,
                generator=generator,
            )

            # Composite the element into the panel
            current = _composite(current, rendered, mask_np[0])

            # ---------------------------------------------------------------
            # Level 3 (optional): fine-detail element rendering
            # ---------------------------------------------------------------
            if elem.type in self.element_level_types:
                current = self._refine_element(
                    current, panel, ei, elem,
                    num_inference_steps, guidance_scale,
                    style_prefix, negative_prompt, generator,
                )

        return current

    def _refine_element(
        self,
        panel_image: Image.Image,
        panel: Panel,
        element_idx: int,
        element: MangaElement,
        num_inference_steps: int,
        guidance_scale: float,
        style_prefix: str = DEFAULT_STYLE_PREFIX,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        generator: torch.Generator | None = None,
    ) -> Image.Image:
        """Level 3: crop an element to full resolution for fine-detail rendering.

        Crops the element's bbox from the panel image, scales to full SD
        resolution, runs a refinement infill pass, then composites back.
        """
        res = self.resolution
        pw, ph = panel_image.size

        # Element bbox in panel pixel coords
        ex1 = max(0, int(element.bbox[0] * pw))
        ey1 = max(0, int(element.bbox[1] * ph))
        ex2 = min(pw, int(element.bbox[2] * pw))
        ey2 = min(ph, int(element.bbox[3] * ph))
        ew, eh = ex2 - ex1, ey2 - ey1
        if ew <= 0 or eh <= 0:
            return panel_image

        # Crop the element at full resolution
        elem_crop = panel_image.crop((ex1, ey1, ex2, ey2))
        elem_crop_res = elem_crop.resize((res, res), Image.LANCZOS)

        # Full mask (refine everything in the crop)
        mask_np = render_full_mask(res)
        mask_pil = Image.fromarray((mask_np[0] * 255).astype(np.uint8), mode="L")

        # Element-level layout
        elem_layout_np = render_element_layout(element, res)

        prompt = element.description
        if element.text:
            prompt += f' with text "{element.text}"'

        refined = self._infill(
            context=elem_crop_res,
            mask=mask_pil,
            layout=elem_layout_np,
            prompt=prompt,
            steps=num_inference_steps,
            guidance=guidance_scale,
            style_prefix=style_prefix,
            negative_prompt=negative_prompt,
            generator=generator,
        )

        # Composite back into panel
        refined_sized = refined.resize((ew, eh), Image.LANCZOS)
        result = panel_image.copy()
        result.paste(refined_sized, (ex1, ey1))
        return result

    # -- Low-level infill call -----------------------------------------------

    def _infill(
        self,
        context: Image.Image,
        mask: Image.Image,
        layout: np.ndarray,
        prompt: str,
        steps: int,
        guidance: float,
        style_prefix: str = DEFAULT_STYLE_PREFIX,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        generator: torch.Generator | None = None,
    ) -> Image.Image:
        """Run one SD inpainting pass with ControlNet layout conditioning.

        Precomputes ControlNet residuals from the layout map, then
        temporarily patches the UNet forward to inject them at every
        denoising step (including both CFG branches).
        """
        import torch.nn.functional as F

        pipe = self.pipe
        device = pipe.device
        dtype = pipe.unet.dtype

        # Prepend style prefix to prompt for consistent style across all calls
        styled_prompt = f"{style_prefix}, {prompt}" if style_prefix else prompt

        # Ensure ControlNet is on the same device/dtype as the UNet
        self.controlnet.to(device=device, dtype=dtype)

        # Precompute ControlNet residuals from layout (pixel → latent space)
        layout_tensor = torch.from_numpy(layout).unsqueeze(0).to(device, dtype=dtype)
        latent_size = self.resolution // 8
        layout_latent = F.interpolate(
            layout_tensor, size=(latent_size, latent_size),
            mode="bilinear", align_corners=False,
        )

        self.controlnet.eval()
        with torch.no_grad():
            down_residuals, mid_residual = self.controlnet(layout_latent)

        # Patch the UNet forward to inject ControlNet residuals each step.
        # During CFG the UNet receives a doubled batch (unconditional +
        # conditional) so we expand the residuals to match.
        original_forward = pipe.unet.forward

        def _forward_with_controlnet(*args, **kwargs):
            sample = args[0] if args else kwargs["sample"]
            B = sample.shape[0]
            kwargs["down_block_additional_residuals"] = [
                r.expand(B, -1, -1, -1) for r in down_residuals
            ]
            kwargs["mid_block_additional_residual"] = mid_residual.expand(
                B, -1, -1, -1
            )
            return original_forward(*args, **kwargs)

        pipe.unet.forward = _forward_with_controlnet
        try:
            result = pipe(
                prompt=styled_prompt,
                negative_prompt=negative_prompt,
                image=context,
                mask_image=mask,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]
        finally:
            pipe.unet.forward = original_forward

        return result


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def _composite(
    canvas: Image.Image,
    result: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """Composite inpainted result into canvas using mask.

    *mask* is (H, W) float in [0, 1].
    """
    canvas_np = np.array(canvas).astype(np.float32)
    result_np = np.array(result.resize(canvas.size, Image.LANCZOS)).astype(np.float32)
    mask_3ch = mask[:, :, None].repeat(3, axis=2)
    composited = canvas_np * (1 - mask_3ch) + result_np * mask_3ch
    return Image.fromarray(composited.astype(np.uint8))
