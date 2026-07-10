"""DDIM inference for Any2RSI with any subset of Canny/HED/Seg controls."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.paths import resolve_anycontrol_root

ANYCONTROL_ROOT = resolve_anycontrol_root()
for path in (ANYCONTROL_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image

from src.train.train import load_state_dict

CONTROL_KEYS = {"canny": "hint_canny", "hed": "hint_hed", "seg": "hint_seg"}


def parse_control(value):
    try:
        control_type, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected TYPE=PATH") from exc
    if control_type not in CONTROL_KEYS:
        raise argparse.ArgumentTypeError(f"unknown control type {control_type!r}")
    return control_type, path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--base-checkpoint", default=None,
        help="initialization checkpoint to load before a trainable-only adapter",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--control", action="append", type=parse_control, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def load_map(path, size, device):
    image = Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.NEAREST
    )
    array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def main():
    args = parse_args()
    if args.size % 64:
        raise SystemExit("--size must be divisible by 64")
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    if args.base_checkpoint:
        load_state_dict(model, args.base_checkpoint)
    load_state_dict(model, args.checkpoint)
    model.eval().cuda()
    device = model.device

    provided = dict(args.control)
    controls, active = [], []
    for control_type in ("canny", "hed", "seg"):
        if control_type in provided:
            controls.append(load_map(provided[control_type], args.size, device))
            active.append(1.0)
        else:
            controls.append(torch.zeros(1, 3, args.size, args.size, device=device))
            active.append(0.0)

    text = model.get_learned_conditioning([args.prompt])
    empty = model.get_learned_conditioning([""])
    shared = dict(
        control_maps=controls,
        control_active=torch.tensor([active], device=device),
        captions=[args.prompt],
    )
    conditioning = dict(c_crossattn=[text], **shared)
    unconditional = dict(c_crossattn=[empty], **shared)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(
        1, 4, args.size // 8, args.size // 8,
        generator=generator, device=device
    )
    sampler = DDIMSampler(model)
    samples, _ = sampler.sample(
        S=args.steps,
        batch_size=1,
        shape=(4, args.size // 8, args.size // 8),
        conditioning=conditioning,
        unconditional_conditioning=unconditional,
        unconditional_guidance_scale=args.cfg_scale,
        x_T=initial_noise,
        verbose=False,
    )
    decoded = model.decode_first_stage(samples)[0]
    decoded = ((decoded.clamp(-1, 1) + 1) * 127.5).byte()
    array = decoded.permute(1, 2, 0).cpu().numpy()
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(destination)
    print(destination)


if __name__ == "__main__":
    main()
