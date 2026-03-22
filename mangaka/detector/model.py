"""
MangaDetectorNet — Hierarchical manga detector with LM description head.

Adapts nissl's RetinaInfillNet architecture:
  Shared ResNet encoder → multi-scale features (f1..f4)
  ├─ Detection head : pool(f4) + prev_bbox → bbox regression + type classification
  └─ Description head: ROI-pooled features → visual prefix → text tokens

The detection head predicts WHAT type of element and WHERE (autoregressive,
teacher-forced). The LM head generates a free-form description conditioned
on visual features extracted from the predicted bounding box region.
Supports GPT-2 and Qwen2.5 (configurable: 0.5B, 1.5B, 3B).

Hierarchy: page → panels → elements (2 levels).

Forward signature:
    forward(img, prev_bbox, description_tokens=None, generate=False)
    → (bbox_pred, type_pred, desc_loss_or_tokens)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import roi_align
from transformers import AutoModelForCausalLM, AutoConfig

from mangaka.schema import NUM_ELEMENT_TYPES
from mangaka.utils.bbox import RETINA_SIZE


# 6 element types + 1 NONE + 1 PANEL = 8 type classes
TYPE_NONE = 0
TYPE_PANEL = 1
TYPE_OFFSET = 2  # element types are TYPE_OFFSET + element_type_id
NUM_TYPE_CLASSES = NUM_ELEMENT_TYPES + TYPE_OFFSET  # 8

_BACKBONE_CONFIGS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
}


class VisualPrefixProjector(nn.Module):
    """Project ROI-pooled visual features into LM embedding space.

    Takes (B, visual_dim) features and produces (B, prefix_length, n_embd)
    prefix tokens that are prepended to the LM input sequence.
    """

    def __init__(self, visual_dim: int, n_embd: int, prefix_length: int):
        super().__init__()
        self.prefix_length = prefix_length
        self.projector = nn.Sequential(
            nn.Linear(visual_dim, n_embd * 2),
            nn.GELU(),
            nn.Linear(n_embd * 2, n_embd * prefix_length),
        )
        self.n_embd = n_embd

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """(B, visual_dim) -> (B, prefix_length, n_embd)."""
        x = self.projector(visual_features)
        return x.view(-1, self.prefix_length, self.n_embd)


class DescriptionHead(nn.Module):
    """LM-based description generator conditioned on visual features.

    Visual features from ROI pooling are projected into prefix tokens that
    condition the causal LM decoder.  During training, teacher-forced with
    ground truth description tokens.  During inference, autoregressive
    generation.  Supports GPT-2 and Qwen2.5 model families.
    """

    def __init__(
        self,
        visual_dim: int,
        lm_model: str = "Qwen/Qwen2.5-0.5B",
        prefix_length: int = 8,
        max_length: int = 64,
    ):
        super().__init__()
        self.prefix_length = prefix_length
        self.max_length = max_length

        # Load pretrained causal LM
        self.lm = AutoModelForCausalLM.from_pretrained(lm_model)
        config = AutoConfig.from_pretrained(lm_model)
        self.n_embd = config.hidden_size
        self.vocab_size = config.vocab_size

        # Visual prefix projector
        self.prefix_proj = VisualPrefixProjector(
            visual_dim=visual_dim,
            n_embd=self.n_embd,
            prefix_length=prefix_length,
        )

        # Type embedding (element type conditions the description)
        self.type_embed = nn.Embedding(NUM_TYPE_CLASSES, self.n_embd)

    def _get_embed_tokens(self):
        """Get the token embedding layer, works for GPT-2 and Qwen."""
        if hasattr(self.lm, 'transformer'):  # GPT-2 style
            return self.lm.transformer.wte
        elif hasattr(self.lm, 'model'):  # Qwen/Llama style
            return self.lm.model.embed_tokens
        raise ValueError(f"Unknown model architecture: {type(self.lm)}")

    def forward(
        self,
        visual_features: torch.Tensor,
        type_ids: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute language modelling loss over description tokens.

        Args:
            visual_features: (B, visual_dim) ROI-pooled features
            type_ids: (B,) element type class ids
            input_ids: (B, seq_len) tokenised descriptions (teacher forcing)
            attention_mask: (B, seq_len) attention mask for input_ids

        Returns:
            loss: scalar cross-entropy loss over description tokens
        """
        B = visual_features.size(0)
        device = visual_features.device

        # Build prefix: visual tokens + type embedding token
        visual_prefix = self.prefix_proj(visual_features)  # (B, P, D)
        type_emb = self.type_embed(type_ids).unsqueeze(1)  # (B, 1, D)
        prefix = torch.cat([visual_prefix, type_emb], dim=1)  # (B, P+1, D)
        prefix_len = prefix.size(1)

        # Embed the text tokens
        embed_tokens = self._get_embed_tokens()
        text_emb = embed_tokens(input_ids)  # (B, S, D)

        # Concatenate prefix + text embeddings
        inputs_embeds = torch.cat([prefix, text_emb], dim=1)  # (B, P+1+S, D)

        # Build attention mask including prefix (always attended)
        prefix_mask = torch.ones(B, prefix_len, dtype=torch.long, device=device)
        if attention_mask is not None:
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            text_mask = torch.ones(B, input_ids.size(1), dtype=torch.long, device=device)
            full_mask = torch.cat([prefix_mask, text_mask], dim=1)

        # Build position ids
        seq_len = inputs_embeds.size(1)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

        # Forward through LM
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits  # (B, P+1+S, vocab)

        # Compute loss only on the text portion (shift by 1 for next-token prediction)
        text_logits = logits[:, prefix_len:-1, :]  # (B, S-1, vocab)
        text_targets = input_ids[:, 1:]  # (B, S-1)

        # Mask out padding positions in loss
        loss = F.cross_entropy(
            text_logits.reshape(-1, self.vocab_size),
            text_targets.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,
        type_ids: torch.Tensor,
        tokenizer,
        max_length: int | None = None,
    ) -> list[str]:
        """Autoregressively generate descriptions.

        Args:
            visual_features: (B, visual_dim)
            type_ids: (B,)
            tokenizer: tokenizer for the LM
            max_length: max tokens to generate

        Returns:
            list of B description strings
        """
        max_length = max_length or self.max_length
        B = visual_features.size(0)
        device = visual_features.device

        # Build prefix
        visual_prefix = self.prefix_proj(visual_features)
        type_emb = self.type_embed(type_ids).unsqueeze(1)
        prefix = torch.cat([visual_prefix, type_emb], dim=1)
        prefix_len = prefix.size(1)

        # Start with BOS token
        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        cur_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        generated = [[] for _ in range(B)]
        eos_id = tokenizer.eos_token_id
        embed_tokens = self._get_embed_tokens()

        for step in range(max_length):
            text_emb = embed_tokens(cur_ids)
            inputs_embeds = torch.cat([prefix, text_emb], dim=1)

            seq_len = inputs_embeds.size(1)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

            outputs = self.lm(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
            )
            next_logits = outputs.logits[:, -1, :]  # (B, vocab)
            next_ids = next_logits.argmax(dim=-1)  # (B,)

            for b in range(B):
                if next_ids[b].item() == eos_id:
                    continue
                generated[b].append(next_ids[b].item())

            cur_ids = torch.cat([cur_ids, next_ids.unsqueeze(1)], dim=1)

            # Stop if all sequences have hit EOS
            if all(
                next_ids[b].item() == eos_id or len(generated[b]) >= max_length
                for b in range(B)
            ):
                break

        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]


class MangaDetectorNet(nn.Module):
    """Hierarchical manga detector with LM description head.

    Shared ResNet encoder feeds two heads:
      1. Detection head — predicts next bbox (x1,y1,x2,y2) + type class
      2. Description head — generates free-form text description via a causal LM
         conditioned on ROI-pooled visual features from the predicted bbox

    Forward signature:
        forward(img, prev_bbox, desc_input_ids=None, desc_attention_mask=None,
                target_bbox_for_roi=None, generate=False, tokenizer=None)
        → (bbox_pred, type_pred, desc_loss_or_texts)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        lm_model: str = "Qwen/Qwen2.5-0.5B",
        prefix_length: int = 8,
        max_description_length: int = 64,
    ):
        super().__init__()

        factory, weights, feat_dim = _BACKBONE_CONFIGS[backbone]
        resnet = factory(weights=weights)
        self.feat_dim = feat_dim

        # --- Shared encoder (from nissl) ---
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ImageNet normalisation buffers
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # --- Detection head ---
        # prev_bbox: (x1, y1, x2, y2, type_id) normalised to [0,1]
        self.fc_shared = nn.Sequential(
            nn.Linear(feat_dim + 5, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )
        self.type_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_TYPE_CLASSES),
        )

        # --- ROI feature extractor for description head ---
        # ROI align from f3 features (64x64 at 1024 input)
        self.roi_output_size = 7
        roi_feat_dim = feat_dim // 2  # f3 channels
        if feat_dim == 512:
            roi_feat_dim = 256  # ResNet-18/34 layer3
        else:
            roi_feat_dim = 1024  # ResNet-50 layer3

        self.roi_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(roi_feat_dim, feat_dim),
            nn.ReLU(),
        )

        # --- Description head (causal LM) ---
        self.description_head = DescriptionHead(
            visual_dim=feat_dim,
            lm_model=lm_model,
            prefix_length=prefix_length,
            max_length=max_description_length,
        )

    def encode(self, img: torch.Tensor):
        """Run the shared encoder, returning multi-scale features."""
        x = (img - self.img_mean) / self.img_std
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4

    def _extract_roi_features(
        self,
        f3: torch.Tensor,
        bboxes_retina: torch.Tensor,
    ) -> torch.Tensor:
        """Extract ROI-pooled features from f3 for given bboxes.

        Args:
            f3: (B, C, H, W) layer3 features
            bboxes_retina: (B, 4) bboxes in retina coordinates [0, RETINA_SIZE]

        Returns:
            (B, feat_dim) ROI-pooled and projected features
        """
        B = bboxes_retina.size(0)
        # Build ROI list: (batch_index, x1, y1, x2, y2) scaled to f3 resolution
        scale = f3.size(2) / RETINA_SIZE
        batch_idx = torch.arange(B, device=f3.device).float().unsqueeze(1)
        rois = torch.cat([batch_idx, bboxes_retina * scale], dim=1)

        # Clamp to valid range
        rois[:, 1:] = rois[:, 1:].clamp(min=0, max=f3.size(2) - 1)
        # Ensure minimum box size for roi_align
        rois[:, 3] = torch.max(rois[:, 3], rois[:, 1] + 1)
        rois[:, 4] = torch.max(rois[:, 4], rois[:, 2] + 1)

        pooled = roi_align(
            f3, rois, output_size=self.roi_output_size,
            spatial_scale=1.0, aligned=True,
        )  # (B, C, 7, 7)

        return self.roi_proj(pooled)  # (B, feat_dim)

    def forward(
        self,
        img: torch.Tensor,
        prev_bbox: torch.Tensor,
        desc_input_ids: torch.Tensor | None = None,
        desc_attention_mask: torch.Tensor | None = None,
        target_bbox_for_roi: torch.Tensor | None = None,
        generate: bool = False,
        tokenizer=None,
    ):
        """
        Args:
            img:               (B, 3, 1024, 1024) retina image, float [0,1]
            prev_bbox:         (B, 5) previous detection (x1,y1,x2,y2,type_id)
                               normalised to [0,1] (bbox) and [0, NUM_TYPE_CLASSES] (type)
            desc_input_ids:    (B, S) tokenised descriptions for teacher forcing
            desc_attention_mask: (B, S) attention mask
            target_bbox_for_roi: (B, 4) bbox in retina coords for ROI pooling
                                 (ground truth during training, predicted during inference)
            generate:          if True, autoregressively generate descriptions
            tokenizer:         required when generate=True

        Returns:
            bbox_pred:  (B, 4) predicted bbox normalised to [0, 1]
            type_pred:  (B, NUM_TYPE_CLASSES) type logits
            desc_out:   description loss (training) or list[str] (generate) or None
        """
        # Shared encoder
        _f1, f2, f3, f4 = self.encode(img)
        pooled = self.pool(f4).view(img.size(0), -1)  # (B, feat_dim)

        # --- Detection head ---
        prev_norm = prev_bbox.clone()
        prev_norm[:, :4] = prev_norm[:, :4] / RETINA_SIZE
        prev_norm[:, 4] = prev_norm[:, 4] / NUM_TYPE_CLASSES

        det = self.fc_shared(torch.cat([pooled, prev_norm], dim=1))
        bbox_pred = self.bbox_head(det)  # (B, 4) in [0, 1]
        type_pred = self.type_head(det)  # (B, NUM_TYPE_CLASSES)

        # --- Description head ---
        desc_out = None

        if desc_input_ids is not None or generate:
            # Determine which bbox to use for ROI extraction
            if target_bbox_for_roi is not None:
                roi_bboxes = target_bbox_for_roi
            else:
                # Use predicted bbox (converted to retina coords)
                roi_bboxes = bbox_pred.detach() * RETINA_SIZE

            # Extract ROI features
            roi_features = self._extract_roi_features(f3, roi_bboxes)

            # Get predicted type ids for conditioning
            type_ids = type_pred.argmax(dim=1)

            if desc_input_ids is not None:
                # Training: compute loss
                desc_out = self.description_head(
                    roi_features, type_ids,
                    desc_input_ids, desc_attention_mask,
                )
            elif generate and tokenizer is not None:
                # Inference: generate descriptions
                desc_out = self.description_head.generate(
                    roi_features, type_ids, tokenizer,
                )

        return bbox_pred, type_pred, desc_out
