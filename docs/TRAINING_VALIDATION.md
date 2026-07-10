# Training validation on RTX 4090

Validated on 2026-07-11 with:

- Base: AnyControl commit `971cef7c340c4054e531b9c5e5e83b74c02f35bc`
- Environment: Python 3.10 (`slurpp`, local validation environment)
- PyTorch 2.4.1 + CUDA 12.1
- GPU: RTX 4090 24GB
- Real Canny, HED and EntitySeg control maps
- AnyControl `init_local.ckpt`, local CLIP-ViT-L and BLIP-2 query initialization

Final smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/smoke_train.py --resolution 512 --enable-pgo
```

Observed result:

```text
SMOKE_TRAIN_OK loss=0.613124 t=107
trainable_tensors_with_finite_grad=474/474
peak_cuda_memory_gib=18.18 pgo=True
train/loss_simple=0.185010
train/loss_vlb=0.001166
train/loss_pgo=0.856228
train/loss=0.613124
```

This verifies model construction, pretrained initialization, VAE/text/control
encoding, CMMCA, 13-level LocalAdapter injection, PGO, backward and AdamW step.

An additional run using a real four-image subset from the local xRST2I-110K
dataset is recorded in `docs/XRST2I_VALIDATION.md`.

The default config uses batch size 1 and gradient accumulation 2. This preserves
an effective batch size of 2 while fitting 512px + PGO on a 24GB 4090.
