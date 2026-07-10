"""Fail-fast validation for an Any2RSI-over-AnyControl checkout."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.paths import resolve_anycontrol_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/any2rsi_rsicd.yaml")
    parser.add_argument("--base-root", default=None)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-annotators", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = OmegaConf.load(config_path)
    params = config.model.params
    base_root = resolve_anycontrol_root(args.base_root)
    errors, warnings = [], []

    required_base_files = [
        "ldm/models/diffusion/ddpm.py",
        "models/local_adapter.py",
        "models/ddim_hacked.py",
    ]
    for filename in required_base_files:
        if not (base_root / filename).is_file():
            errors.append(f"missing AnyControl base file: {base_root / filename}")
    expected_commit_file = PROJECT_ROOT / "third_party" / "ANYCONTROL_COMMIT"
    if expected_commit_file.is_file() and (base_root / ".git").exists():
        expected_commit = expected_commit_file.read_text(encoding="utf-8").strip()
        result = subprocess.run(
            ["git", "-C", str(base_root), "rev-parse", "HEAD"],
            text=True, capture_output=True,
        )
        if result.returncode or result.stdout.strip() != expected_commit:
            errors.append(
                f"AnyControl commit mismatch: expected {expected_commit}, "
                f"found {result.stdout.strip() or 'unknown'}"
            )
    if not (PROJECT_ROOT / "src/models/any2rsi_cldm.py").is_file():
        errors.append("missing overlay file: src/models/any2rsi_cldm.py")

    checkpoint = PROJECT_ROOT / "weights/anycontrol/ckpts/init_local.ckpt"
    if not checkpoint.is_file() or checkpoint.stat().st_size != 5913265211:
        errors.append(f"missing or incomplete init checkpoint: {checkpoint}")
    clip_root = Path(str(params.vision_encoder_config.params.model_name))
    if not clip_root.is_absolute():
        clip_root = PROJECT_ROOT / clip_root
    clip_files = {
        "model.safetensors": 1710540580,
        "config.json": 4519,
        "tokenizer.json": 2224003,
    }
    for filename, expected_size in clip_files.items():
        path = clip_root / filename
        if not path.is_file() or path.stat().st_size != expected_size:
            errors.append(f"missing or incomplete CLIP file: {path}")
    qformer_file = PROJECT_ROOT / "weights/anycontrol/ckpts/blip2_pretrained.pth"
    if not qformer_file.is_file() or qformer_file.stat().st_size != 746998955:
        errors.append(f"missing or incomplete BLIP-2 checkpoint: {qformer_file}")
    if not args.skip_annotators:
        annotators = {
            "network-bsds500.pth": 58871680,
            "CropFormer_hornet_3x_03823a.pth": 888996425,
        }
        for filename, expected_size in annotators.items():
            path = base_root / "annotator" / "ckpts" / filename
            if not path.is_file() or path.stat().st_size != expected_size:
                errors.append(f"missing or incomplete annotator checkpoint: {path}")

    required_modules = [
        "torch", "transformers", "omegaconf", "pytorch_lightning", "cv2"
    ]
    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            errors.append(f"missing Python dependency: {module}")

    if params.unet_config.target != "models.local_adapter.LocalControlUNetModel":
        errors.append("unet_config must target AnyControl LocalControlUNetModel")
    query_count = int(params.cmmca_config.params.num_query_tokens)
    side = int(query_count ** 0.5)
    if side * side != query_count:
        errors.append("CMMCA num_query_tokens must be a perfect square")
    if len(params.control_keys) != len(config.data.params.train.params.control_types):
        errors.append("model control_keys and dataset control_types disagree")

    if not args.skip_data:
        data = config.data.params.train.params
        for key in ("images_dir", "enriched_json", "controls_dir"):
            value = Path(str(data[key]))
            if not value.is_absolute():
                value = PROJECT_ROOT / value
            if not value.exists():
                errors.append(f"configured data path does not exist: {key}={value}")
        controls_root = Path(str(data.controls_dir))
        if not controls_root.is_absolute():
            controls_root = PROJECT_ROOT / controls_root
        for control_type in data.control_types:
            if not (controls_root / control_type).is_dir():
                errors.append(
                    f"missing precomputed control directory: "
                    f"{controls_root / control_type}"
                )

    qformer_ckpt = params.cmmca_config.params.get("qformer_init_ckpt")
    if not qformer_ckpt:
        warnings.append(
            "qformer_init_ckpt is unset; query tokens will be randomly initialized"
        )
    else:
        qformer_path = Path(str(qformer_ckpt))
        if not qformer_path.is_absolute():
            qformer_path = PROJECT_ROOT / qformer_path
        if not qformer_path.is_file():
            errors.append(f"missing Q-Former checkpoint: {qformer_path}")

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Any2RSI setup checks passed")


if __name__ == "__main__":
    main()
