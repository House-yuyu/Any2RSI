"""Prepare an xRST2I image tree and caption manifest without redistributing it."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="xRST2I_110K root")
    parser.add_argument("--out", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=None,
                        help="optional deterministic subset size")
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"],
                        default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    image_root = root / args.split / args.split
    text_root = root / f"{args.split}_Refined_txt"
    if not image_root.is_dir() or not text_root.is_dir():
        raise SystemExit(
            f"expected {image_root} and {text_root}; see docs/DATASET.md"
        )

    out = Path(args.out).expanduser().resolve()
    caption_path = out / "enriched.json"
    if caption_path.exists() and not args.overwrite:
        raise SystemExit(f"{caption_path} exists; pass --overwrite to replace it")

    pairs = []
    for text_path in sorted(text_root.rglob("*.txt")):
        relative_stem = text_path.relative_to(text_root).with_suffix("")
        image_path = next(
            (image_root / relative_stem.with_suffix(ext) for ext in IMAGE_EXTENSIONS
             if (image_root / relative_stem.with_suffix(ext)).is_file()),
            None,
        )
        if image_path is not None:
            pairs.append((image_path, text_path))
        if args.count is not None and len(pairs) >= args.count:
            break
    if not pairs:
        raise SystemExit("no matched image/text pairs found")

    captions = {}
    for index, (image_path, text_path) in enumerate(pairs, 1):
        relative = image_path.relative_to(image_root)
        materialize(image_path, out / "images" / relative, args.mode)
        captions[relative.as_posix()] = text_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        if index % 1000 == 0:
            print(f"prepared {index}/{len(pairs)}")

    out.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(
        json.dumps(captions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {len(captions)} pairs at {out}")
    print("Run scripts/prepare_controls.py next; refined captions can be used as-is")
    print("or enriched offline with scripts/edg_generate.py.")


if __name__ == "__main__":
    main()
