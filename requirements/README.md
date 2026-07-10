# Dependency groups

- `base.txt`: validated training/inference stack; PyTorch must match the local
  CUDA driver/toolkit.
- `annotators.txt`: source-built detectron2 required by EntitySeg.
- `edg.txt`: optional VLM/API enriched-description backends.
- `eval.txt`: optional clean-FID implementation.
- `dev.txt`: unit tests and linting.

`environment.yml` is the recommended reproducible starting point. AnyControl is
pinned separately as a Git submodule because it is an application dependency,
not a normal PyPI package. EntitySeg's MSDeformAttn extension must be compiled
after the environment and AnyControl checkout are ready.
