# Local model weights (not tracked)

Run `python scripts/download_weights.py` from the repository root. Expected
paths are documented and verified by `configs/weights.json`.

Large model files in this directory are intentionally ignored by Git. Do not
force-add them: the AnyControl initialization checkpoint is larger than normal
GitHub and Git LFS per-file limits.
