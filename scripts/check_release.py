"""Fail if common private-machine or GitHub release hazards are present."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".toml", ".txt", ".sh"}
PRIVATE_PATTERNS = [
    re.compile(r"/data\d*/users/"),
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
]


def tracked_files() -> list[Path]:
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=ROOT, check=True, text=True, capture_output=True,
        )
        return [ROOT / line for line in result.stdout.splitlines() if line]
    excluded = {"weights", "data", "logs", "outputs", "smoke_data", "xrst2i_subset"}
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and not any(part in excluded for part in path.relative_to(ROOT).parts)
    ]


def main() -> None:
    errors = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.is_symlink() and not path.exists():
            errors.append(f"broken symlink: {relative} -> {os.readlink(path)}")
            continue
        if path.is_file() and path.stat().st_size > 99 * 1024 * 1024:
            errors.append(f"file exceeds 99 MiB: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in PRIVATE_PATTERNS:
                if pattern.search(content):
                    errors.append(f"private path/token pattern in {relative}: {pattern.pattern}")

    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Release hygiene checks passed")


if __name__ == "__main__":
    main()
