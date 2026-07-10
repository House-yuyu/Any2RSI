from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_anycontrol_root(explicit: str | os.PathLike | None = None) -> Path:
    """Return the first configured AnyControl checkout.

    Search order: explicit CLI argument, environment variable, vendored/pinned
    checkout under ``third_party``, then a legacy sibling checkout.
    """
    root = project_root()
    candidates = [
        explicit,
        os.environ.get("ANY2RSI_ANYCONTROL_ROOT"),
        root / "third_party" / "AnyControl",
        root.parent / "AnyControl",
    ]
    resolved = [Path(value).expanduser().resolve() for value in candidates if value]
    for candidate in resolved:
        if (candidate / "ldm" / "models" / "diffusion" / "ddpm.py").is_file():
            return candidate
    return resolved[0] if resolved else root / "third_party" / "AnyControl"
