"""Export only trainable Any2RSI tensors from a full Lightning checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.paths import resolve_anycontrol_root

ANYCONTROL_ROOT = resolve_anycontrol_root()
if str(ANYCONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYCONTROL_ROOT))

from ldm.util import instantiate_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/any2rsi_rsicd.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = instantiate_from_config(OmegaConf.load(args.config).model)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=False)
    trainable_names = {name for name, parameter in model.named_parameters()
                       if parameter.requires_grad}
    exported = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                if name in trainable_names}
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": exported,
        "metadata": {
            "format": "any2rsi-trainable-v1",
            "source_checkpoint": Path(args.checkpoint).name,
            "base_checkpoint_required": True,
            "tensor_count": len(exported),
        },
    }, destination)
    size_mib = destination.stat().st_size / 1024 ** 2
    print(f"exported {len(exported)} tensors to {destination} ({size_mib:.1f} MiB)")


if __name__ == "__main__":
    main()
