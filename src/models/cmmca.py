"""
src/models/cmmca.py

Cross-Modal Multi-Control Adapter (CMMCA) -- the Any2RSI remote-sensing variant of
AnyControl's Multi-Control Encoder.

Relationship to AnyControl:
  AnyControl's Multi-Control Encoder uses alternating *multi-control fusion* and
  *multi-control alignment* blocks united by query tokens. Any2RSI keeps that
  alternating structure but renames/reframes the two blocks as:
    - Cross-Modal Context Aggregator  (Eq.3 : self-attn over [Q, Te])
    - Multi-Control Cross Attention   (Eq.4,5 : Q cross-attends to all V_l + P)
  and adds a shared learnable position embedding P over vision tokens plus a
  task-specific text token to bridge the modality gap.

This module is registered via instantiate_from_config and is consumed by the
LatentDiffusion subclass (see src/models/any2rsi_cldm.py). Its multi-level query
grids are projected by AnyControl's LocalAdapter and injected spatially into the
SD UNet.

Designed to be a drop-in target in the YAML:

    control_stage_config:
      target: src.models.cmmca.CMMCA
      params:
        dim: 768
        ...
"""
from __future__ import annotations

import torch
import torch.nn as nn

# AnyControl / ControlNet ship a memory-efficient attention; we fall back to
# torch's MultiheadAttention so this file works standalone if xformers is absent.
try:
    import xformers  # noqa: F401
    import xformers.ops as xops
    _HAS_XFORMERS = True
except Exception:
    _HAS_XFORMERS = False


def exists(x):
    return x is not None


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention supporting self- and cross-attention.

    Uses xformers memory-efficient attention when available (matches AnyControl
    runtime behavior), else falls back to nn.MultiheadAttention.
    """

    def __init__(self, dim, n_heads=8, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def _split(self, x, B):
        # [B, N, D] -> [B, H, N, hd]
        N = x.shape[1]
        return x.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, context=None, key_mask=None):
        """Apply attention.

        ``key_mask`` is a boolean tensor shaped ``[B, Nk]`` where ``True``
        marks a valid key.  This is important for arbitrary control subsets:
        encoding a zero image with CLIP does *not* produce zero tokens.
        """
        context = x if context is None else context
        B = x.shape[0]
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        use_xformers = (
            _HAS_XFORMERS
            and key_mask is None
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        if use_xformers:
            # xformers expects [B, N, H, hd]
            def reshape(t):
                N = t.shape[1]
                return t.view(B, N, self.n_heads, self.head_dim)
            try:
                out = xops.memory_efficient_attention(
                    reshape(q), reshape(k), reshape(v))
                out = out.reshape(B, -1, self.n_heads * self.head_dim)
            except (NotImplementedError, RuntimeError):
                # Wheels often expose xFormers on machines where a particular
                # device, dtype or head size has no compiled kernel.
                use_xformers = False
        if not use_xformers:
            qh, kh, vh = self._split(q, B), self._split(k, B), self._split(v, B)
            attn = (qh @ kh.transpose(-1, -2)) * self.scale
            if key_mask is not None:
                if key_mask.shape != (B, context.shape[1]):
                    raise ValueError(
                        "key_mask must have shape [batch, context_tokens], got "
                        f"{tuple(key_mask.shape)} for context {tuple(context.shape)}"
                    )
                valid = key_mask.to(device=attn.device, dtype=torch.bool)
                # Every dataset sample is required to retain at least one
                # control, but keep the attention numerically safe as well.
                no_valid_key = ~valid.any(dim=1)
                if no_valid_key.any():
                    valid = valid.clone()
                    valid[no_valid_key, 0] = True
                attn = attn.masked_fill(~valid[:, None, None, :],
                                        torch.finfo(attn.dtype).min)
            attn = attn.softmax(dim=-1)
            out = attn @ vh                       # [B,H,N,hd]
            out = out.transpose(1, 2).reshape(B, -1, self.n_heads * self.head_dim)
        return self.to_out(out)


class CrossModalContextAggregator(nn.Module):
    """Eq.3 -- self-attention over concatenated [Q, Te] (the 'fusion' block)."""

    def __init__(self, dim, n_heads=8, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, q, te):
        nq = q.shape[1]
        x = torch.cat([q, te], dim=1)
        x = x + self.attn(self.norm(x))
        x = x + self.ff(self.norm_ff(x))
        return x[:, :nq], x[:, nq:]


class MultiControlCrossAttention(nn.Module):
    """Eq.4,5 -- Q cross-attends to concat of all controls' vision tokens (+ P).

    V'_l = V_l + P  ;  Q = CrossAtt(Q, [V'_1, ..., V'_n])  (the 'alignment' block).
    """

    def __init__(self, dim, n_heads=8, dropout=0.0, max_tokens_per_ctrl=257):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens_per_ctrl, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, q, vision_tokens, control_mask=None):
        """
        vision_tokens: list of n tensors [B, Tv, D]
        control_mask:  optional [B, n] (1=present, 0=absent).  It is expanded
                       over every vision token so absent controls cannot
                       participate in cross-attention.
        """
        v_pos = []
        for v in vision_tokens:
            tv = v.shape[1]
            if tv > self.pos.shape[1]:
                raise ValueError(
                    f"vision token count {tv} exceeds max_tokens_per_ctrl="
                    f"{self.pos.shape[1]}"
                )
            v_pos.append(v + self.pos[:, :tv, :])
        v_cat = torch.cat(v_pos, dim=1)
        key_mask = None
        if control_mask is not None:
            if control_mask.ndim != 2 or control_mask.shape[1] != len(v_pos):
                raise ValueError(
                    "control_mask must have shape [batch, num_controls], got "
                    f"{tuple(control_mask.shape)} for {len(v_pos)} controls"
                )
            key_mask = torch.cat([
                control_mask[:, i:i + 1].bool().expand(-1, v.shape[1])
                for i, v in enumerate(v_pos)
            ], dim=1)
        q = q + self.attn(self.norm_q(q), context=self.norm_v(v_cat),
                          key_mask=key_mask)
        q = q + self.ff(self.norm_ff(q))
        return q


class CMMCA(nn.Module):
    def __init__(self, dim=768, text_dim=768, vision_dim=1024,
                 num_query_tokens=256, num_layers=4, n_heads=8, dropout=0.0,
                 out_context_dim=768, qformer_init_ckpt: str | None = None):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if num_query_tokens < 1:
            raise ValueError("num_query_tokens must be at least 1")
        self.num_query_tokens = num_query_tokens
        self.dim = dim

        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.task_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.task_token, std=0.02)

        self.text_proj = nn.Linear(text_dim, dim)
        self.vision_proj = nn.Linear(vision_dim, dim)

        self.aggregators = nn.ModuleList(
            [CrossModalContextAggregator(dim, n_heads, dropout)
             for _ in range(num_layers)])
        self.cross_attns = nn.ModuleList(
            [MultiControlCrossAttention(dim, n_heads, dropout)
             for _ in range(num_layers)])

        self.out_proj = nn.Linear(dim, out_context_dim)
        self.out_norm = nn.LayerNorm(out_context_dim)

        if qformer_init_ckpt:
            self._load_qformer_init(qformer_init_ckpt)

    @torch.no_grad()
    def _load_qformer_init(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for container_key in ("state_dict", "model"):
            if isinstance(sd, dict) and isinstance(sd.get(container_key), dict):
                sd = sd[container_key]
                break
        # BLIP-2 stores query tokens under 'query_tokens'; shapes may differ.
        key = next((k for k in sd if "query_token" in k), None)
        if key is None:
            return
        w = sd[key]
        if w.dim() == 2:
            w = w.unsqueeze(0)
        nq = min(self.num_query_tokens, w.shape[1])
        d = min(self.dim, w.shape[2])
        self.query_tokens.data[:, :nq, :d] = w[:, :nq, :d].to(
            self.query_tokens.dtype)

    def forward(self, text_emb, control_vision_tokens, control_mask=None,
                return_hidden_states=False):
        """
        Args:
            text_emb: [B, Lt, text_dim]  (enriched CLIP text embedding, Te)
            control_vision_tokens: list of n tensors [B, Tv, vision_dim]
            control_mask: optional [B, n]
        Returns:
            context: [B, num_query_tokens, out_context_dim]
        """
        B = text_emb.shape[0]
        te = self.text_proj(text_emb)
        te = torch.cat([self.task_token.expand(B, -1, -1), te], dim=1)
        q = self.query_tokens.expand(B, -1, -1).contiguous()
        vtoks = [self.vision_proj(v) for v in control_vision_tokens]

        hidden_states = []
        for agg, xa in zip(self.aggregators, self.cross_attns):
            q, te = agg(q, te)
            q = xa(q, vtoks, control_mask)
            hidden_states.append(self.out_norm(self.out_proj(q)))
        if return_hidden_states:
            return hidden_states
        return hidden_states[-1]
