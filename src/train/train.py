"""
src/train/train.py

PyTorch Lightning entrypoint for Any2RSI. Loads the OmegaConf YAML, instantiates
the Any2RSIControlLDM model and the data module, and runs trainer.fit.

Usage (single GPU):
    python src/train/train.py --config configs/any2rsi_rsicd.yaml \
        --resume ckpts/init_local.ckpt

Usage (multi-GPU, AnyControl-style):
    python -m torch.distributed.launch --nproc_per_node 8 src/train/train.py \
        --config configs/any2rsi_rsicd.yaml --learning-rate 1e-5 \
        --batch-size 8 --training-steps 90000

This mirrors AnyControl's src/train/train.py contract (config-path / batch-size /
training-steps / log-freq) while remaining compatible with a plain config flag.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.paths import resolve_anycontrol_root

ANYCONTROL_ROOT = resolve_anycontrol_root()
for path in (ANYCONTROL_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytorch_lightning as pl
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


class DataModuleFromConfig(pl.LightningDataModule):
    """Minimal data module: builds the train Dataset from config and serves a
    DataLoader. Mirrors the ControlNet/LDM DataModuleFromConfig contract."""

    def __init__(self, batch_size, train, num_workers=4, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_cfg = train
        self._train_ds = None

    def setup(self, stage=None):
        self._train_ds = instantiate_from_config(self.train_cfg)

    def train_dataloader(self):
        return DataLoader(self._train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          drop_last=True, pin_memory=True)


def load_state_dict(model, ckpt_path):
    import torch
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[resume] loaded {ckpt_path} | missing {len(missing)} "
          f"| unexpected {len(unexpected)}")
    return model


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "--config-path", dest="config", required=True)
    ap.add_argument("--resume", "--resume-path", dest="resume", default=None,
                    help="ckpt to warm-start (e.g. ckpts/init_local.ckpt)")
    ap.add_argument("--adapter", default=None,
                    help="trainable-only adapter to load after --resume")
    ap.add_argument("--resume-training", default=None,
                    help="full Lightning checkpoint restoring optimizer/step")
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--training-steps", type=int, default=None)
    ap.add_argument("--log-freq", type=int, default=None)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--logdir", default="logs/any2rsi")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--captions", default=None)
    ap.add_argument("--controls-dir", default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--precision", default=None)
    ap.add_argument("--disable-pgo", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    if args.resume_training and (args.resume or args.adapter):
        raise SystemExit(
            "--resume-training is mutually exclusive with --resume/--adapter"
        )
    pl.seed_everything(args.seed, workers=True)
    cfg = OmegaConf.load(args.config)

    # CLI overrides matching AnyControl's flags
    if args.learning_rate is not None:
        cfg.model.base_learning_rate = args.learning_rate
    if args.batch_size is not None:
        cfg.data.params.batch_size = args.batch_size
    if args.training_steps is not None:
        cfg.lightning.trainer.max_steps = args.training_steps
    if args.log_freq is not None:
        cfg.lightning.callbacks.image_logger.batch_frequency = args.log_freq
    train_params = cfg.data.params.train.params
    if args.images_dir is not None:
        train_params.images_dir = args.images_dir
    if args.captions is not None:
        train_params.enriched_json = args.captions
    if args.controls_dir is not None:
        train_params.controls_dir = args.controls_dir
    if args.image_size is not None:
        train_params.image_size = args.image_size
    if args.num_workers is not None:
        cfg.data.params.num_workers = args.num_workers
    if args.precision is not None:
        cfg.lightning.trainer.precision = args.precision
    if args.disable_pgo:
        cfg.model.params.pgo_config.enable = False

    model = instantiate_from_config(cfg.model)
    model.learning_rate = cfg.model.base_learning_rate
    if args.resume:
        load_state_dict(model, args.resume)
    if args.adapter:
        load_state_dict(model, args.adapter)

    data = instantiate_from_config(cfg.data)

    callbacks = [ModelCheckpoint(every_n_train_steps=(
        args.log_freq or cfg.lightning.callbacks.image_logger.batch_frequency
    ), save_last=True)]
    try:
        from models.logger import ImageLogger
        callbacks.append(ImageLogger(
            batch_frequency=cfg.lightning.callbacks.image_logger.batch_frequency))
    except Exception:
        pass  # image logger optional

    tcfg = cfg.lightning.trainer
    trainer = pl.Trainer(
        max_steps=tcfg.get("max_steps", 40000),
        accumulate_grad_batches=tcfg.get("accumulate_grad_batches", 1),
        precision=tcfg.get("precision", 32),
        gradient_clip_val=tcfg.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        default_root_dir=args.logdir,
        accelerator="gpu", devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
    )
    trainer.fit(model, data, ckpt_path=args.resume_training)


if __name__ == "__main__":
    main()
