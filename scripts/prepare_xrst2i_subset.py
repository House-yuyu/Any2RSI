"""Create a deterministic, symlink-based subset of xRST2I-110K."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    image_root = root / "train" / "train"
    text_root = root / "train_Refined_txt"
    out = Path(args.out).resolve()
    out_images = out / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    matched = []
    for text_path in sorted(text_root.glob("*.txt")):
        image_path = next(
            (image_root / f"{text_path.stem}{ext}" for ext in IMAGE_EXTENSIONS
             if (image_root / f"{text_path.stem}{ext}").is_file()),
            None,
        )
        if image_path is not None:
            matched.append((image_path, text_path))
        if len(matched) == args.count:
            break
    if len(matched) < args.count:
        raise SystemExit(f"only found {len(matched)} matched image-text pairs")

    captions = {}
    for image_path, text_path in matched:
        destination = out_images / image_path.name
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(image_path)
        captions[image_path.name] = text_path.read_text(encoding="utf-8").strip()
    (out / "captions.json").write_text(
        json.dumps(captions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {len(captions)} pairs at {out}")


if __name__ == "__main__":
    main()
