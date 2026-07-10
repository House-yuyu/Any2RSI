# Architecture and data flow

```text
coarse text + image + scene type
               |
               v
       EDG enriched description ----------------------+
                                                       |
Canny / HED / segmentation                             v
          |                                  frozen CLIP text encoder
          v                                            |
  multi-level CLIP vision tokens                       |
          |                                            |
          +---------------> CMMCA <--------------------+
                              |
                   four 16x16 query states
                              |
                   AnyControl LocalAdapter
                              |
              13 multi-scale residual features
                              |
noise latent + text --------> SD 1.5 UNet --------> predicted x0
                                                       |
                                             differentiable VAE + CLIP
                                                       |
                                                    PGO loss
```

EDG is offline and does not receive gradients. During training, a random
non-empty subset of control maps is active. CMMCA masks inactive controls before
cross-attention, then maps its per-layer query states into the spatial feature
hierarchy expected by AnyControl's LocalAdapter. The diffusion objective and
weighted PGO objective update CMMCA and LocalAdapter while SD, VAE, visual CLIP,
text CLIP, and PGO CLIP remain frozen in the default configuration.

See `docs/IMPLEMENTATION_DETAILS.md` for the module-level parameterization.
