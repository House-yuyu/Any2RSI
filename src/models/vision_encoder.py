"""
src/models/vision_encoder.py

Frozen multi-level vision encoder over spatial control maps. Extracts vision
tokens from several layers of a pretrained ViT (the paper's multi-level strategy
for RS multi-scale structure). One control map (HED/Canny/Seg/...) is encoded at
a time; the CMMCA fuses across controls.

instantiate_from_config target:
    target: src.models.vision_encoder.MultiLevelVisionEncoder
    params:
      model_name: openai/clip-vit-large-patch14
      layer_indices: [6, 12, 18, 24]
      tokens_per_step: 1
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPVisionModel
    _HAS_HF = True
except Exception:
    _HAS_HF = False


class MultiLevelVisionEncoder(nn.Module):
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_name="openai/clip-vit-large-patch14",
                 layer_indices=(6, 12, 18, 24), tokens_per_step=1,
                 input_size=224, freeze=True):
        super().__init__()
        if not _HAS_HF:
            raise ImportError("transformers required for MultiLevelVisionEncoder")
        self.vit = CLIPVisionModel.from_pretrained(model_name)
        self.hidden = self.vit.config.hidden_size
        self.layer_indices = tuple(layer_indices)
        self.tokens_per_step = tokens_per_step
        self.input_size = input_size
        self.freeze = freeze
        if freeze:
            self.vit.eval()
            for p in self.vit.parameters():
                p.requires_grad_(False)
        self.register_buffer("_mean",
                             torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1), False)
        self.register_buffer("_std",
                             torch.tensor(self.CLIP_STD).view(1, 3, 1, 1), False)

    @property
    def output_dim(self):
        return self.hidden

    def _preprocess(self, x):
        # x in [-1,1] -> CLIP-normalized at input_size
        x = (x.clamp(-1, 1) + 1) / 2
        x = F.interpolate(x, size=self.input_size, mode="bilinear",
                          align_corners=False)
        return (x - self._mean) / self._std

    @torch.no_grad()
    def forward(self, control_map, training=True):
        """control_map: [B,3,H,W] in [-1,1] -> [B, k*Tv, hidden]."""
        pix = self._preprocess(control_map)
        out = self.vit(pixel_values=pix, output_hidden_states=True)
        hs = out.hidden_states                       # tuple len n_layers+1
        n = len(hs)
        cand = [i for i in self.layer_indices if i < n]
        if not cand:
            raise ValueError(
                f"none of layer_indices={self.layer_indices} exists in a "
                f"vision encoder exposing {n} hidden states"
            )
        take = min(self.tokens_per_step, len(cand))
        if training:
            # torch RNG follows Lightning/DDP seeding; Python's global random
            # module does not reliably do so in every worker/process.
            order = torch.randperm(len(cand), device=control_map.device)[:take]
            chosen = sorted(cand[i] for i in order.cpu().tolist())
        else:
            chosen = sorted(cand[-take:])
        return torch.cat([hs[i] for i in chosen], dim=1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.vit.eval()
        return self
