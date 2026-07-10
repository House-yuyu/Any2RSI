# Installation and operation guide

All commands below run from the Any2RSI repository root. Paths are intentionally
portable; no personal-machine path is required.

## 1. Create the environment

The validated stack is Python 3.10, PyTorch 2.4.1, CUDA 12.1, torchvision
0.19.1, PyTorch Lightning 2.5.1, and Transformers 4.51.3.

```bash
conda env create -f environment.yml
conda activate any2rsi
```

For a pip-managed environment, install the PyTorch wheel matching your CUDA
runtime first, then run `pip install -r requirements/base.txt`.

## 2. Install the pinned AnyControl base

```bash
bash scripts/setup_anycontrol.sh
```

The script clones AnyControl into `third_party/AnyControl` and checks out the
commit stored in `third_party/ANYCONTROL_COMMIT`. To use an existing checkout:

```bash
export ANY2RSI_ANYCONTROL_ROOT=/path/to/AnyControl
```

Install and compile the optional HED/EntitySeg preprocessing stack:

```bash
pip install -r requirements/annotators.txt
cd third_party/AnyControl/annotator/entityseg/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd -
```

EntitySeg is legacy code. `prepare_controls.py` contains the Pillow compatibility
shim used in the validated environment. A C++ compiler and CUDA toolkit matching
PyTorch are required to compile MSDeformAttn.

## 3. Download weights

Review `THIRD_PARTY_NOTICES.md` and the upstream model cards first, then run:

```bash
python scripts/download_weights.py
bash scripts/setup_anycontrol.sh
```

The second setup call links annotator weights into AnyControl. To validate files
without network access:

```bash
python scripts/download_weights.py --verify-only
```

Expected sizes and SHA256 values are in `configs/weights.json`. No weight is
tracked by Git.

## 4. Prepare data and EDG captions

See `docs/DATASET.md`. A typical xRST2I workflow is:

```bash
python scripts/prepare_xrst2i.py --root /path/to/xRST2I_110K --out data
python scripts/edg_generate.py \
  --images_dir data/images \
  --coarse_captions data/enriched.json \
  --out data/edg.json --backend internvl
python scripts/prepare_controls.py \
  --images-dir data/images --out-dir data/conditions --types canny hed seg
```

If EDG output is used for training, set `data.params.train.params.enriched_json`
in the YAML to `data/edg.json`. EDG resumes from the existing JSON by default;
failed images are written to a separate failure file.

## 5. Validate and test

```bash
python scripts/check_setup.py
python -m pytest -q
python scripts/check_release.py
```

To run a GPU optimizer-step smoke test using a local fixture:

```bash
python scripts/make_smoke_data.py --size 512
python scripts/smoke_train.py --resolution 512 --enable-pgo
```

PGO retains gradients through the frozen VAE and CLIP image encoder. The
validated peak allocation at 512×512 and batch size 1 was 18.18 GiB on an RTX
4090. Lowering PGO probability/timestep coverage changes the experiment.

## 6. Training and resume modes

Warm-start from AnyControl:

```bash
python src/train/train.py \
  --config configs/any2rsi_rsicd.yaml \
  --resume weights/anycontrol/ckpts/init_local.ckpt \
  --gpus 1 --training-steps 40000
```

Resume optimizer, scheduler, step, and model from a full Lightning checkpoint:

```bash
python src/train/train.py \
  --config configs/any2rsi_rsicd.yaml \
  --resume-training logs/any2rsi/checkpoints/last.ckpt --gpus 1
```

Warm-start from the base plus an exported adapter:

```bash
python src/train/train.py \
  --config configs/any2rsi_rsicd.yaml \
  --resume weights/anycontrol/ckpts/init_local.ckpt \
  --adapter /path/to/any2rsi_adapter.ckpt --gpus 1
```

## 7. Inference and evaluation

`scripts/generate.py` accepts any non-empty subset of `canny`, `hed`, and `seg`
through repeated `--control TYPE=PATH` arguments. Use
`scripts/evaluate.py --help` for CLIP similarity, paired PSNR/SSIM, and optional
clean-FID.

For an evaluation split, `scripts/generate_batch.py` loads the model once and
generates every caption whose requested controls exist. Its `seed + sample_index`
rule makes each output deterministic for a fixed sorted manifest.

## Troubleshooting

- **LatentDiffusion not found:** run `setup_anycontrol.sh` or set
  `ANY2RSI_ANYCONTROL_ROOT`.
- **MSDeformAttn import error:** compile the EntitySeg extension inside the active
  environment and ensure `nvcc` matches the PyTorch CUDA ABI.
- **CUDA out of memory:** first verify batch size 1 and gradient accumulation 2;
  then disable PGO only for pipeline diagnosis.
- **Hash mismatch:** delete only the affected local file and rerun the downloader.
- **Git rejects a file:** never force-add weights; publish adapters in a model hub.
