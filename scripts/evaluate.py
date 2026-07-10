"""Transparent baseline evaluation for generated remote-sensing images.

This does not claim to reproduce the paper's private evaluation protocol. It
computes reproducible CLIP similarity, paired PSNR/SSIM, and optional clean-FID.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import CLIPModel, CLIPProcessor

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def image_map(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.suffix.lower() in EXTENSIONS
    }


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--captions", required=True,
                        help="JSON mapping relative image names to prompts")
    parser.add_argument("--clip-model", default="weights/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    generated_root = Path(args.generated)
    generated = image_map(generated_root)
    captions = json.loads(Path(args.captions).read_text(encoding="utf-8"))
    names = sorted(set(generated) & set(captions))
    if not names:
        raise SystemExit("no generated images match caption keys")

    model = CLIPModel.from_pretrained(args.clip_model).eval().to(args.device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    similarities = []
    for start in range(0, len(names), args.batch_size):
        batch_names = names[start:start + args.batch_size]
        images = [Image.open(generated[name]).convert("RGB") for name in batch_names]
        inputs = processor(
            text=[captions[name] for name in batch_names], images=images,
            return_tensors="pt", padding=True, truncation=True,
        ).to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
            text_features = torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
        similarities.extend((image_features * text_features).sum(-1).cpu().tolist())

    report = {
        "num_generated": len(generated),
        "num_caption_matched": len(names),
        "clip_cosine_mean": float(np.mean(similarities)),
        "clip_score_x100": float(100 * np.mean(np.maximum(similarities, 0))),
        "clip_model": args.clip_model,
    }

    if args.reference:
        reference_root = Path(args.reference)
        reference = image_map(reference_root)
        paired = sorted(set(names) & set(reference))
        psnr_values, ssim_values = [], []
        for name in paired:
            target = load_rgb(reference[name])
            prediction = np.asarray(
                Image.open(generated[name]).convert("RGB").resize(
                    (target.shape[1], target.shape[0]), Image.Resampling.BICUBIC
                )
            )
            psnr_values.append(peak_signal_noise_ratio(target, prediction, data_range=255))
            ssim_values.append(structural_similarity(
                target, prediction, channel_axis=2, data_range=255
            ))
        report.update(
            num_paired=len(paired),
            psnr_mean=float(np.mean(psnr_values)) if paired else None,
            ssim_mean=float(np.mean(ssim_values)) if paired else None,
        )
        if args.fid:
            try:
                from cleanfid import fid
            except ImportError as exc:
                raise SystemExit("install requirements/eval.txt for --fid") from exc
            report["clean_fid"] = float(fid.compute_fid(
                str(generated_root), str(reference_root)
            ))

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
