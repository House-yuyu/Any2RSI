"""Run one real Any2RSI optimizer step on a tiny local fixture dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/any2rsi_rsicd.yaml")
    parser.add_argument("--base-root", default=None)
    parser.add_argument(
        "--checkpoint", default="weights/anycontrol/ckpts/init_local.ckpt"
    )
    parser.add_argument(
        "--clip", default="weights/clip-vit-large-patch14"
    )
    parser.add_argument("--data-root", default="smoke_data")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--enable-pgo", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)
    from src.utils.paths import resolve_anycontrol_root

    base_root = str(resolve_anycontrol_root(args.base_root))
    if base_root not in sys.path:
        sys.path.append(base_root)

    # Import only after AnyControl's ldm/ and models/ are visible.
    from ldm.util import instantiate_from_config

    from src.train.train import load_state_dict

    cfg = OmegaConf.load(args.config)
    clip_path = str(Path(args.clip).resolve())
    data_root = Path(args.data_root).resolve()
    cfg.model.params.vision_encoder_config.params.model_name = clip_path
    cfg.model.params.pgo_config.clip_model = clip_path
    cfg.model.params.cond_stage_config.params.version = clip_path
    cfg.model.params.pgo_config.enable = bool(args.enable_pgo)
    cfg.data.params.batch_size = 1
    cfg.data.params.num_workers = 0
    train = cfg.data.params.train.params
    train.images_dir = str(data_root / "images")
    train.enriched_json = str(data_root / "captions.json")
    train.controls_dir = str(data_root / "conditions")
    train.image_size = args.resolution
    train.min_controls = 3
    train.max_controls = 3
    train.drop_text_prob = 0.0
    train.drop_all_prob = 0.0

    torch.manual_seed(23)
    model = instantiate_from_config(cfg.model)
    model.learning_rate = cfg.model.base_learning_rate
    load_state_dict(model, args.checkpoint)
    model.train().cuda()
    optimizer = model.configure_optimizers()
    dataset = instantiate_from_config(cfg.data.params.train)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    x_start, cond = model.get_input(batch, model.first_stage_key)
    t = torch.randint(0, model.num_timesteps, (1,), device=model.device)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, metrics = model.p_losses(x_start, cond, t)
    loss.backward()
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    grad_count = sum(p.grad is not None and torch.isfinite(p.grad).all()
                     for p in trainable)
    optimizer.step()
    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"SMOKE_TRAIN_OK loss={loss.item():.6f} t={t.item()} ")
    print(f"trainable_tensors_with_finite_grad={grad_count}/{len(trainable)}")
    print(f"peak_cuda_memory_gib={peak_gib:.2f} pgo={args.enable_pgo}")
    for key, value in metrics.items():
        print(f"{key}={float(value):.6f}")


if __name__ == "__main__":
    main()
