"""Any2RSI model integration for the official AnyControl codebase.

The original overlay bypassed AnyControl's spatial path and called the vanilla
UNet with an enlarged text context.  That makes the edge/segmentation maps lose
their spatial correspondence.  This implementation keeps AnyControl's actual
``LocalAdapter -> LocalControlUNetModel`` path and replaces its Q-Former output
with the paper's CMMCA multi-level query features.
"""
from __future__ import annotations

import importlib
import math
import os

import torch
import torch.nn.functional as F
from ldm.util import default, instantiate_from_config
from torch.utils.checkpoint import checkpoint


def _resolve_base():
    """Resolve the LatentDiffusion shipped by AnyControl/ControlNet."""
    candidates = [
        os.environ.get("ANY2RSI_LATENT_DIFFUSION", ""),
        "ldm.models.diffusion.ddpm.LatentDiffusion",
        "src.ldm.models.diffusion.ddpm.LatentDiffusion",
    ]
    errors = []
    for path in candidates:
        if not path:
            continue
        module_name, _, class_name = path.rpartition(".")
        try:
            return getattr(importlib.import_module(module_name), class_name)
        except Exception as exc:  # retain diagnostics for an actionable error
            errors.append(f"{path}: {exc}")
    raise ImportError(
        "Could not locate LatentDiffusion. Overlay this package on the official "
        "open-mmlab/AnyControl repository, or set ANY2RSI_LATENT_DIFFUSION. "
        + " | ".join(errors)
    )


BaseLatentDiffusion = _resolve_base()


class Any2RSIControlLDM(BaseLatentDiffusion):
    """Stable Diffusion v1.5 with CMMCA, AnyControl spatial injection and PGO."""

    def __init__(
        self,
        cmmca_config,
        vision_encoder_config,
        local_control_config,
        control_keys=("hint_canny", "hint_hed", "hint_seg"),
        pgo_config=None,
        sd_locked=True,
        local_control_scales=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cmmca = instantiate_from_config(cmmca_config)
        self.vision_encoder = instantiate_from_config(vision_encoder_config)
        self.local_adapter = instantiate_from_config(local_control_config)
        self.control_keys = list(control_keys)
        self.sd_locked = bool(sd_locked)
        self.local_control_scales = local_control_scales or [1.0] * 13

        # AnyControl freezes the SD input/middle path and learns its adapter.
        self.model.diffusion_model.requires_grad_(False)
        if not self.sd_locked:
            self.model.diffusion_model.output_blocks.requires_grad_(True)
            self.model.diffusion_model.out.requires_grad_(True)

        pgo_config = pgo_config or {}
        self.pgo_enable = bool(pgo_config.get("enable", True))
        self.pgo_weight = float(pgo_config.get("weight", 0.5))
        self.pgo_prob = float(pgo_config.get("prob", 1.0))
        self.pgo_max_t_frac = float(pgo_config.get("max_t_frac", 1.0))
        self.pgo_gradient_checkpointing = bool(
            pgo_config.get("gradient_checkpointing", True)
        )
        if not 0 <= self.pgo_prob <= 1 or not 0 <= self.pgo_max_t_frac <= 1:
            raise ValueError("PGO prob and max_t_frac must lie in [0, 1]")
        if self.pgo_enable:
            self._init_pgo_clip(
                pgo_config.get("clip_model", "openai/clip-vit-large-patch14")
            )

    # ------------------------------------------------------------------ PGO
    def _init_pgo_clip(self, name):
        from transformers import CLIPModel, CLIPTokenizer

        self.pgo_clip = CLIPModel.from_pretrained(name)
        self.pgo_tokenizer = CLIPTokenizer.from_pretrained(name)
        self.pgo_clip.eval().requires_grad_(False)
        if self.pgo_gradient_checkpointing:
            self.pgo_clip.gradient_checkpointing_enable()

    def train(self, mode=True):
        super().train(mode)
        # Frozen perceptual encoders must not switch dropout/training behavior.
        self.vision_encoder.eval()
        if self.pgo_enable:
            self.pgo_clip.eval()
        return self

    def _decode_for_pgo(self, latent):
        """Differentiable VAE decode.

        LatentDiffusion.decode_first_stage is decorated with ``no_grad``.  PGO
        needs gradients with respect to the denoiser output, while the frozen VAE
        parameters themselves remain excluded from the optimizer.
        """
        scaled = latent / self.scale_factor
        if self.training and self.pgo_gradient_checkpointing:
            return checkpoint(
                self.first_stage_model.decode, scaled, use_reentrant=False
            )
        return self.first_stage_model.decode(scaled)

    def _pgo_loss(self, pred_x0, t, captions):
        """Equation (2): CLIP distance between a denoised estimate and caption."""
        batch_size = pred_x0.shape[0]
        eligible = (t.float() / float(self.num_timesteps)) <= self.pgo_max_t_frac
        if self.pgo_prob < 1.0:
            eligible &= torch.rand(batch_size, device=t.device) < self.pgo_prob
        indices = eligible.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            return pred_x0.new_zeros(())

        imgs = self._decode_for_pgo(pred_x0.index_select(0, indices))
        imgs = (imgs.clamp(-1, 1) + 1) / 2
        imgs = F.interpolate(imgs, size=224, mode="bicubic", align_corners=False)
        mean = imgs.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(
            1, 3, 1, 1
        )
        std = imgs.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(
            1, 3, 1, 1
        )
        image_features = self.pgo_clip.get_image_features(
            pixel_values=(imgs - mean) / std
        )

        if captions is None:
            captions = [""] * batch_size
        selected_captions = [captions[i] for i in indices.cpu().tolist()]
        tokens = self.pgo_tokenizer(
            selected_captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(pred_x0.device)
        with torch.no_grad():
            text_features = self.pgo_clip.get_text_features(**tokens)

        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        return (1.0 - (image_features * text_features).sum(-1)).mean()

    def _pred_x0(self, x_t, model_output, t):
        if self.parameterization == "eps":
            alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sigma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            return (x_t - sigma * model_output) / alpha
        if self.parameterization == "x0":
            return model_output
        if self.parameterization == "v":
            return self.predict_start_from_z_and_v(x_t, t, model_output)
        raise NotImplementedError(self.parameterization)

    # --------------------------------------------------------------- input
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, text_condition = super().get_input(
            batch, k, bs=bs, *args, **kwargs
        )
        controls = []
        for key in self.control_keys:
            if key not in batch:
                raise KeyError(f"training batch is missing control key {key!r}")
            control = torch.as_tensor(batch[key], device=self.device).float()
            if control.ndim == 4 and control.shape[-1] in (1, 3):
                control = control.permute(0, 3, 1, 2).contiguous()
            if bs is not None:
                control = control[:bs]
            controls.append(control)

        active = batch.get("active")
        if active is None:
            active = torch.ones(
                x.shape[0], len(controls), device=self.device, dtype=x.dtype
            )
        else:
            active = torch.as_tensor(active, device=self.device).float()
            if bs is not None:
                active = active[:bs]

        captions = batch.get("caption", batch.get("txt"))
        if bs is not None and captions is not None:
            captions = captions[:bs]
        if isinstance(text_condition, dict):
            cond = dict(text_condition)
        else:
            cond = {"c_crossattn": [text_condition]}
        cond.update(
            control_maps=controls,
            control_active=active,
            captions=captions,
        )
        return x, cond

    # ------------------------------------------------------- spatial control
    def _queries_to_local_features(self, query_states):
        """Project four 16x16 CMMCA query grids to AnyControl feature scales."""
        projections = self.local_adapter.visual_projs
        scales = self.local_adapter.query_scales
        if len(query_states) != len(projections):
            # Evenly select levels if a custom CMMCA depth is used.
            positions = torch.linspace(
                0, len(query_states) - 1, len(projections)
            ).round().long().tolist()
            query_states = [query_states[i] for i in positions]

        features = []
        for query, scale, projection in zip(query_states, scales, projections):
            batch, token_count, channels = query.shape
            side = math.isqrt(token_count)
            if side * side != token_count:
                raise ValueError(
                    "CMMCA query count must form a square spatial grid; got "
                    f"{token_count} tokens"
                )
            spatial = query.transpose(1, 2).reshape(batch, channels, side, side)
            spatial = F.interpolate(
                spatial,
                scale_factor=float(scale),
                mode="bilinear",
                align_corners=False,
            )
            height, width = spatial.shape[-2:]
            tokens = spatial.flatten(2).transpose(1, 2)
            tokens = projection(tokens)
            features.append(
                tokens.transpose(1, 2).reshape(batch, -1, height, width)
            )
        return features

    def _build_local_features(self, cond):
        text_embedding = torch.cat(cond["c_crossattn"], dim=1)
        vision_tokens = [
            self.vision_encoder(control, training=self.training)
            for control in cond["control_maps"]
        ]
        query_states = self.cmmca(
            text_embedding,
            vision_tokens,
            cond.get("control_active"),
            return_hidden_states=True,
        )
        local_features = self._queries_to_local_features(query_states)
        active = cond.get("control_active")
        if active is not None:
            has_control = active.bool().any(dim=1).to(text_embedding.dtype)
            local_features = [
                feature * has_control[:, None, None, None]
                for feature in local_features
            ]
        return text_embedding, local_features

    def apply_model(self, x_noisy, t, cond, local_strength=1.0, *args, **kwargs):
        if not isinstance(cond, dict):
            raise TypeError("Any2RSI conditioning must be a dictionary")
        text_embedding, local_features = self._build_local_features(cond)
        local_control = self.local_adapter(
            x=x_noisy,
            timesteps=t,
            context=text_embedding,
            local_features=local_features,
        )
        if len(local_control) != len(self.local_control_scales):
            raise ValueError(
                f"LocalAdapter returned {len(local_control)} controls, but "
                f"{len(self.local_control_scales)} scales were configured"
            )
        local_control = [
            feature * scale
            for feature, scale in zip(local_control, self.local_control_scales)
        ]
        try:
            return self.model.diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=text_embedding,
                local_control=local_control,
                local_w=local_strength,
            )
        except TypeError as exc:
            raise TypeError(
                "The configured UNet must be models.local_adapter."
                "LocalControlUNetModel from the official AnyControl repository"
            ) from exc

    # -------------------------------------------------------------- losses
    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(self.parameterization)

        prefix = "train" if self.training else "val"
        loss_dict = {}
        loss_simple = self.get_loss(model_output, target, mean=False).mean(
            [1, 2, 3]
        )
        loss_dict[f"{prefix}/loss_simple"] = loss_simple.mean().detach()
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict[f"{prefix}/loss_gamma"] = loss.mean().detach()
            loss_dict["logvar"] = self.logvar.data.mean()
        loss = self.l_simple_weight * loss.mean()
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(
            dim=(1, 2, 3)
        )
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss = loss + self.original_elbo_weight * loss_vlb
        loss_dict[f"{prefix}/loss_vlb"] = loss_vlb.detach()

        if self.pgo_enable:
            pred_x0 = self._pred_x0(x_noisy, model_output, t)
            pgo = self._pgo_loss(pred_x0, t, cond.get("captions"))
            loss = loss + self.pgo_weight * pgo
            loss_dict[f"{prefix}/loss_pgo"] = pgo.detach()
        loss_dict[f"{prefix}/loss"] = loss.detach()
        return loss, loss_dict

    def configure_optimizers(self):
        parameters = list(self.cmmca.parameters()) + list(
            self.local_adapter.parameters()
        )
        parameters += [
            parameter
            for parameter in self.vision_encoder.parameters()
            if parameter.requires_grad
        ]
        if not self.sd_locked:
            parameters += list(self.model.diffusion_model.output_blocks.parameters())
            parameters += list(self.model.diffusion_model.out.parameters())
        return torch.optim.AdamW(
            parameters, lr=self.learning_rate, weight_decay=1e-2
        )
