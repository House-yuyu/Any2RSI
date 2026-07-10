"""Generate an evaluation directory while loading the model only once."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.paths import resolve_anycontrol_root

ANYCONTROL_ROOT = resolve_anycontrol_root()
if str(ANYCONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYCONTROL_ROOT))

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from scripts.generate import CONTROL_KEYS, load_map
from src.train.train import load_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/any2rsi_rsicd.yaml")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--controls-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--control-types", nargs="+", choices=list(CONTROL_KEYS),
                        default=list(CONTROL_KEYS))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.size % 64:
        raise SystemExit("--size must be divisible by 64")

    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    if args.base_checkpoint:
        load_state_dict(model, args.base_checkpoint)
    load_state_dict(model, args.checkpoint)
    model.eval().cuda()
    sampler = DDIMSampler(model)
    device = model.device

    captions = json.loads(Path(args.captions).read_text(encoding="utf-8"))
    controls_root = Path(args.controls_dir)
    output_root = Path(args.out_dir)
    generated, skipped = 0, 0
    for index, (name, caption) in enumerate(sorted(captions.items())):
        if isinstance(caption, list):
            caption = caption[0]
        destination = output_root / name
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue

        controls, active = [], []
        for control_type in CONTROL_KEYS:
            source = controls_root / control_type / name
            enabled = control_type in args.control_types and source.is_file()
            if enabled:
                controls.append(load_map(source, args.size, device))
                active.append(1.0)
            else:
                controls.append(torch.zeros(
                    1, 3, args.size, args.size, device=device
                ))
                active.append(0.0)
        if not any(active):
            print(f"skip {name}: no requested control map")
            skipped += 1
            continue

        text = model.get_learned_conditioning([caption])
        empty = model.get_learned_conditioning([""])
        shared = {
            "control_maps": controls,
            "control_active": torch.tensor([active], device=device),
            "captions": [caption],
        }
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        noise = torch.randn(
            1, 4, args.size // 8, args.size // 8,
            generator=generator, device=device,
        )
        with torch.no_grad():
            samples, _ = sampler.sample(
                S=args.steps,
                batch_size=1,
                shape=(4, args.size // 8, args.size // 8),
                conditioning={"c_crossattn": [text], **shared},
                unconditional_conditioning={"c_crossattn": [empty], **shared},
                unconditional_guidance_scale=args.cfg_scale,
                x_T=noise,
                verbose=False,
            )
            decoded = model.decode_first_stage(samples)[0]
        array = ((decoded.clamp(-1, 1) + 1) * 127.5).byte()
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array.permute(1, 2, 0).cpu().numpy()).save(destination)
        generated += 1
        print(f"{generated} generated: {destination}")
    print(f"done: generated={generated}, skipped={skipped}, out={output_root}")


if __name__ == "__main__":
    main()
