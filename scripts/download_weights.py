"""Download and verify third-party weights without adding them to Git."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify(entry: dict) -> tuple[bool, str]:
    path = ROOT / entry["destination"]
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size != entry["size"]:
        return False, f"size={path.stat().st_size}, expected={entry['size']}"
    actual = sha256(path)
    if actual != entry["sha256"]:
        return False, f"sha256={actual}, expected={entry['sha256']}"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/weights.json")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-annotators", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    entries = manifest["files"]
    if args.skip_annotators:
        entries = [e for e in entries if "annotator/" not in e["filename"]]
    if args.skip_clip:
        entries = [e for e in entries if e["repo"] != manifest["clip_repo"]]

    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise SystemExit("install huggingface-hub before downloading") from exc

        for entry in entries:
            destination = ROOT / entry["destination"]
            valid = destination.is_file() and destination.stat().st_size == entry["size"]
            if valid and not args.force:
                print(f"present {destination.relative_to(ROOT)}")
                continue
            local_root = ROOT / (
                "weights/clip-vit-large-patch14"
                if entry["repo"] == manifest["clip_repo"]
                else "weights/anycontrol"
            )
            if entry["repo"] == manifest["clip_repo"]:
                print(f"downloading repository {entry['repo']} -> {local_root}")
                snapshot_download(
                    repo_id=entry["repo"],
                    local_dir=local_root,
                    ignore_patterns=["*.bin"],
                )
            else:
                print(f"downloading {entry['repo']}:{entry['filename']}")
                hf_hub_download(
                    repo_id=entry["repo"],
                    filename=entry["filename"],
                    local_dir=local_root,
                    force_download=args.force,
                )

    failures = []
    for entry in entries:
        ok, message = verify(entry)
        print(f"{'OK' if ok else 'FAIL'} {entry['destination']}: {message}")
        if not ok:
            failures.append(entry["destination"])
    if failures:
        raise SystemExit(f"weight verification failed for {len(failures)} file(s)")
    print("All selected weights verified")


if __name__ == "__main__":
    main()
