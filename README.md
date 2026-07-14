# :fire: Any2RSI: Controllable Remote Sensing Text-to-Image Generation via Any Control and Enriched Description (AAAI 2026)

> [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/38283)] [[DOI](https://doi.org/10.1609/aaai.v40i15.38283)]

This is the official PyTorch implementation for our paper:

> **Any2RSI: Controllable Remote Sensing Text-to-Image Generation via Any Control and Enriched Description**<br>
> [Xu Zhang](https://house-yuyu.github.io/), Jianzhong Huang, [Lefei Zhang](https://scholar.google.com.hk/citations?user=BLKHwNwAAAAJ&hl=en)<br>
> School of Computer Science, Wuhan University

:star: If Any2RSI is helpful to your research or projects, please consider starring this repository. Thank you!

## :sparkles: Highlights

- **Any-control remote-sensing generation.** Any2RSI accepts arbitrary combinations of Canny, HED, and segmentation conditions for flexible spatial control.
- **VLM-Empowered Enriched Description Generation.** EDG enriches coarse remote-sensing captions with objects, layouts, relationships, scene context, and visual attributes.
- **Cross-Modal Multi-Control Adapter.** CMMCA aligns enriched text with multi-level visual control tokens and produces four query feature levels for spatial injection.
- **Prompt-Guided Optimization.** PGO applies a differentiable CLIP image-text objective to improve semantic consistency between generated images and descriptions.

## :hammer_and_wrench: Environment

Recommended environment:

```text
Linux
Python 3.10
PyTorch 2.4.1
CUDA 12.1
```

Clone the repository and create the environment:

```bash
git clone --recursive https://github.com/House-yuyu/Any2RSI.git
cd Any2RSI

conda env create -f environment.yml
conda activate any2rsi
```

Prepare the pinned AnyControl base and annotators:

```bash
bash scripts/setup_anycontrol.sh
pip install -r requirements/annotators.txt

cd third_party/AnyControl/annotator/entityseg/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd -
```

Detailed installation and troubleshooting are available in [docs/INSTALL.md](docs/INSTALL.md).

## Directory Structure

```text
Any2RSI/
├── configs/
│   ├── any2rsi_rsicd.yaml
│   └── weights.json
├── data/
│   └── README.md
├── docs/
├── requirements/
├── scripts/
│   ├── download_weights.py
│   ├── edg_generate.py
│   ├── prepare_xrst2i.py
│   ├── prepare_controls.py
│   ├── smoke_train.py
│   ├── generate.py
│   ├── generate_batch.py
│   ├── evaluate.py
│   └── export_trainable.py
├── src/
│   ├── data/
│   ├── models/
│   │   ├── any2rsi_cldm.py
│   │   ├── cmmca.py
│   │   └── vision_encoder.py
│   └── train/
├── tests/
├── third_party/
│   └── AnyControl/             # pinned Git submodule
├── weights/                    # generated locally, ignored by Git
├── environment.yml
└── README.md
```

## :floppy_disk: Weight Preparation

Download CLIP, AnyControl initialization, BLIP-2 query initialization, HED, and EntitySeg weights:

```bash
python scripts/download_weights.py
bash scripts/setup_anycontrol.sh
```

Verify file sizes and SHA256 checksums:

```bash
python scripts/download_weights.py --verify-only
python scripts/check_setup.py --skip-data
```

The downloaded files are placed under `weights/` and linked to the corresponding AnyControl annotator paths.

## :open_file_folder: Data Preparation

The training data should use the following layout:

```text
data/
├── images/
│   └── <relative-image-path>
├── enriched.json
└── conditions/
    ├── canny/
    │   └── <relative-image-path>
    ├── hed/
    │   └── <relative-image-path>
    └── seg/
        └── <relative-image-path>
```

Prepare xRST2I image-text pairs:

```bash
python scripts/prepare_xrst2i.py \
  --root /path/to/xRST2I_110K \
  --out data
```

Generate enriched descriptions with InternVL2.5:

```bash
python scripts/edg_generate.py \
  --images_dir data/images \
  --coarse_captions data/enriched.json \
  --out data/edg.json \
  --backend internvl
```

Generate the three spatial conditions:

```bash
python scripts/prepare_controls.py \
  --images-dir data/images \
  --out-dir data/conditions \
  --types canny hed seg
```

Set `enriched_json: data/edg.json` in the training config when using the EDG output. See [docs/DATASET.md](docs/DATASET.md) for the manifest format and preprocessing details.

## :rocket: Training

Single-GPU training:

```bash
python src/train/train.py \
  --config configs/any2rsi_rsicd.yaml \
  --resume weights/anycontrol/ckpts/init_local.ckpt \
  --training-steps 40000 \
  --gpus 1
```

The default config uses `512x512` images, AdamW with a learning rate of `1e-5`, physical batch size 1, and gradient accumulation 2.

Resume a complete Lightning training state:

```bash
python src/train/train.py \
  --config configs/any2rsi_rsicd.yaml \
  --resume-training logs/any2rsi/checkpoints/last.ckpt \
  --gpus 1
```

Run a synthetic optimizer-step smoke test:

```bash
python scripts/make_smoke_data.py --size 512
python scripts/smoke_train.py --resolution 512 --enable-pgo
```

## :art: Inference

Export the trainable Any2RSI parameters from a Lightning checkpoint:

```bash
python scripts/export_trainable.py \
  --checkpoint logs/any2rsi/checkpoints/last.ckpt \
  --out outputs/any2rsi_adapter.ckpt
```

Generate one remote-sensing image with any subset of controls:

```bash
python scripts/generate.py \
  --config configs/any2rsi_rsicd.yaml \
  --base-checkpoint weights/anycontrol/ckpts/init_local.ckpt \
  --checkpoint outputs/any2rsi_adapter.ckpt \
  --prompt "an aerial view of an airport with parallel runways" \
  --control canny=/path/to/canny.png \
  --control seg=/path/to/seg.png \
  --steps 50 \
  --cfg-scale 7.5 \
  --out outputs/airport.png
```

Generate a complete evaluation split:

```bash
python scripts/generate_batch.py \
  --config configs/any2rsi_rsicd.yaml \
  --base-checkpoint weights/anycontrol/ckpts/init_local.ckpt \
  --checkpoint outputs/any2rsi_adapter.ckpt \
  --captions data/enriched.json \
  --controls-dir data/conditions \
  --out-dir outputs/eval
```

## :chart_with_upwards_trend: Evaluation

Evaluate CLIP similarity and paired PSNR/SSIM, with optional clean-FID:

```bash
pip install -r requirements/eval.txt

python scripts/evaluate.py \
  --generated outputs/eval \
  --reference data/images \
  --captions data/enriched.json \
  --clip-model weights/clip-vit-large-patch14 \
  --fid \
  --out outputs/eval_metrics.json
```

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

## :handshake: Acknowledgements

This project builds on [AnyControl](https://github.com/open-mmlab/AnyControl), Stable Diffusion v1.5, OpenAI CLIP, BLIP-2/LAVIS, HED, and EntitySeg. We thank the authors and open-source contributors for their excellent work.

## :postbox: Contact

If you have any questions, please feel free to contact us at zhangx0802@whu.edu.cn.
