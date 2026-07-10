"""Precompute the three spatial controls used by Any2RSI.

This script must be run from an official AnyControl checkout after the overlay
has been applied and annotator checkpoints have been installed.
"""
from __future__ import annotations

import argparse
import os
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

import cv2
import numpy as np


def build_processors(control_types):
    processors = {}
    if "canny" in control_types:
        from annotator.canny import CannyDetector

        processors["canny"] = CannyDetector()
    if "hed" in control_types:
        from annotator.hed import HEDdetector

        processors["hed"] = HEDdetector()
    if "seg" in control_types:
        # EntitySeg imports its local mask2former package as a top-level module.
        from PIL import Image

        # detectron2 0.6 predates Pillow 10's enum cleanup.
        if not hasattr(Image, "LINEAR"):
            Image.LINEAR = Image.Resampling.BILINEAR
        entity_root = str(ANYCONTROL_ROOT / "annotator" / "entityseg")
        if entity_root not in sys.path:
            sys.path.insert(0, entity_root)
        from annotator.entityseg import EntitysegDetector

        previous_cwd = os.getcwd()
        try:
            os.chdir(ANYCONTROL_ROOT)
            processors["seg"] = EntitysegDetector()
        finally:
            os.chdir(previous_cwd)
    return processors


def hwc3(image):
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--types", nargs="+", default=["canny", "hed", "seg"],
        choices=["canny", "hed", "seg"]
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    processors = build_processors(args.types)
    root = Path(args.images_dir)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    paths = sorted(path for path in root.rglob("*")
                   if path.suffix.lower() in extensions)
    if not paths:
        raise SystemExit(f"no images found under {root}")

    failures = []
    for index, path in enumerate(paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(str(path))
            continue
        relative = path.relative_to(root)
        for control_type, processor in processors.items():
            destination = Path(args.out_dir) / control_type / relative
            if destination.exists() and not args.overwrite:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            result = hwc3(processor(image))
            if not cv2.imwrite(str(destination), result):
                failures.append(str(destination))
        if index % 100 == 0 or index == len(paths):
            print(f"{index}/{len(paths)}")

    if failures:
        failure_file = Path(args.out_dir) / "failures.txt"
        failure_file.parent.mkdir(parents=True, exist_ok=True)
        failure_file.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} failures; see {failure_file}")


if __name__ == "__main__":
    main()
