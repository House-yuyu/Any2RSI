# Any2RSI: Any Control and Enriched Description for Remote-Sensing T2I

Official implementation of **Any2RSI: Controllable Remote Sensing Text-to-Image
Generation via Any Control and Enriched Description** (AAAI 2026). It supports
arbitrary combinations of Canny, HED, and segmentation controls with enriched
remote-sensing descriptions.

## What is implemented

- EDG with InternVL2.5, Qwen2.5-VL, and a dependency-free pipeline backend;
- CMMCA with active-control masking and multi-level visual tokens;
- AnyControl `LocalAdapter` spatial injection through 13 UNet residuals;
- differentiable CLIP prompt-guided optimization (PGO);
- xRST2I preparation, real Canny/HED/EntitySeg preprocessing, training,
  checkpoint resume, adapter export, DDIM inference, and baseline evaluation;
- unit tests, release hygiene checks, weight checksums, and GitHub Actions.

## Quick start

Prerequisites: Linux, Python 3.10, CUDA 12.1-compatible NVIDIA driver, at least
24 GB VRAM for 512×512 training with PGO, and enough disk space for roughly
10 GB of prerequisite weights.

```bash
conda env create -f environment.yml
conda activate any2rsi
bash scripts/setup_anycontrol.sh
pip install -r requirements/annotators.txt
python scripts/download_weights.py
bash scripts/setup_anycontrol.sh
python scripts/check_setup.py --skip-data
python -m pytest -q
```

Prepare local xRST2I data (not included):

```bash
python scripts/prepare_xrst2i.py --root /path/to/xRST2I_110K --out data
python scripts/prepare_controls.py \
  --images-dir data/images --out-dir data/conditions --types canny hed seg
python scripts/check_setup.py
```

Train:

```bash
python src/train/train.py \
  --config configs/any2rsi_rsicd.yaml \
  --resume weights/anycontrol/ckpts/init_local.ckpt \
  --training-steps 40000 --gpus 1
```

Export a smaller adapter and generate:

```bash
python scripts/export_trainable.py \
  --checkpoint logs/any2rsi/checkpoints/last.ckpt \
  --out outputs/any2rsi_adapter.ckpt

python scripts/generate.py \
  --config configs/any2rsi_rsicd.yaml \
  --base-checkpoint weights/anycontrol/ckpts/init_local.ckpt \
  --checkpoint outputs/any2rsi_adapter.ckpt \
  --prompt "an aerial view of an airport with parallel runways" \
  --control canny=/path/to/canny.png \
  --control seg=/path/to/seg.png \
  --out outputs/airport.png
```

Full installation and troubleshooting are in [docs/INSTALL.md](docs/INSTALL.md).
The model/data flow is summarized in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Validation

| Capability | Status |
|---|---|
| Model construction and pretrained initialization | validated |
| Canny/HED/EntitySeg arbitrary subsets | validated |
| 512×512 forward/backward/AdamW with PGO | validated on RTX 4090 |
| xRST2I sample training | validated |



## Citation

```bibtex
@article{zhang2026any2rsi,
  title   = {Any2RSI: Controllable Remote Sensing Text-to-Image Generation via Any Control and Enriched Description},
  author  = {Zhang, Xu and Huang, Jianzhong and Zhang, Lefei},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {15},
  pages   = {12852--12860},
  year    = {2026},
  doi     = {10.1609/aaai.v40i15.38283}
}
```
