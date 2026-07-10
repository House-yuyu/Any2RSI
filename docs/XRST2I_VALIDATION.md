# xRST2I-110K subset validation

Source dataset: a locally obtained xRST2I-110K copy; source files are not
distributed by this repository.

Dataset inspection found exactly 108,000 training images and 108,000 refined
text files. A deterministic four-pair subset was created at `xrst2i_subset/`
using `scripts/prepare_xrst2i_subset.py`. The original 113GB dataset was not
modified.

Real Canny, HED and EntitySeg maps were generated for all four images. The final
512x512 PGO-enabled optimizer step used one of these real remote-sensing pairs:

```text
SMOKE_TRAIN_OK loss=0.909503 t=107
trainable_tensors_with_finite_grad=474/474
peak_cuda_memory_gib=18.18 pgo=True
train/loss_simple=0.536821
train/loss_vlb=0.003384
train/loss_pgo=0.745363
train/loss=0.909503
```

With the configured PGO weight 0.5, the total is consistent with
`loss_ldm + 0.5 * loss_pgo` (up to the configured VLB term). This run completed
forward, backward and AdamW update successfully.
