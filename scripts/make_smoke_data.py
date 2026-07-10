"""Create a small synthetic, redistributable fixture for pipeline smoke tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="smoke_data")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if args.size < 64 or args.size % 64:
        raise SystemExit("--size must be >=64 and divisible by 64")

    root = Path(args.out)
    name = "synthetic_airport.png"
    for relative in ["images", "conditions/canny", "conditions/hed", "conditions/seg"]:
        (root / relative).mkdir(parents=True, exist_ok=True)

    size = args.size
    image = Image.new("RGB", (size, size), (74, 102, 62))
    draw = ImageDraw.Draw(image)
    runway_width = max(8, size // 24)
    for y in (size // 3, 2 * size // 3):
        draw.rectangle((size // 10, y - runway_width, 9 * size // 10, y + runway_width),
                       fill=(118, 118, 116))
        for x in range(size // 8, 7 * size // 8, max(12, size // 16)):
            draw.rectangle((x, y - 1, x + size // 32, y + 1), fill=(235, 235, 220))
    draw.rectangle((size // 2, size // 3, size // 2 + runway_width,
                    2 * size // 3), fill=(102, 104, 103))
    image.save(root / "images" / name)

    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES).convert("RGB")
    edges.save(root / "conditions" / "canny" / name)
    edges.filter(ImageFilter.GaussianBlur(radius=max(1, size / 512))).save(
        root / "conditions" / "hed" / name
    )

    segmentation = Image.new("RGB", (size, size), (34, 139, 34))
    seg_draw = ImageDraw.Draw(segmentation)
    for y in (size // 3, 2 * size // 3):
        seg_draw.rectangle(
            (size // 10, y - runway_width, 9 * size // 10, y + runway_width),
            fill=(128, 128, 128),
        )
    seg_draw.rectangle((size // 2, size // 3, size // 2 + runway_width,
                        2 * size // 3), fill=(128, 128, 128))
    segmentation.save(root / "conditions" / "seg" / name)

    (root / "captions.json").write_text(json.dumps({
        name: "A synthetic aerial airport with two parallel runways, a central taxiway, and surrounding grass."
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Synthetic fixture created at {root.resolve()}")


if __name__ == "__main__":
    main()
