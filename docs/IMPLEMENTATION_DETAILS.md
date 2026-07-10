# Implementation details

1. **CMMCA initialization.** The BLIP-2 checkpoint initializes the first 32 of
   256 learnable query tokens. The remaining query tokens use truncated-normal
   initialization. CMMCA alternates cross-modal context aggregation and
   multi-control cross-attention for four layers.
2. **Vision hierarchy.** Multi-level tokens come from CLIP ViT-L/14 hidden
   layers 6, 12, 18, and 24.
3. **Task token.** A learned task token is appended to the embedded enriched-text
   sequence before CMMCA aggregation.
4. **PGO target.** PGO differentiably decodes predicted `x0` and applies a
   weighted CLIP image-text distance. The default weight is 0.5.
5. **Spatial adapter.** Four CMMCA query grids feed AnyControl's LocalAdapter and
   its 13 residual injection points in the diffusion UNet.
6. **EDG backends.** The offline EDG pipeline supports InternVL2.5-8B,
   Qwen2.5-VL, an OpenAI-compatible multimodal API, and an echo backend for
   pipeline checks.
7. **Evaluation.** The evaluator provides CLIP similarity, paired PSNR/SSIM, and
   optional clean-FID with explicit model and data paths.

Experiment reports should identify the repository commit, checkpoint, dataset
split, model revision, configuration, seed, and evaluation protocol.
