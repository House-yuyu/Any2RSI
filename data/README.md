# Local datasets (not tracked)

Expected training layout:

```text
data/
  images/<relative-image-path>
  enriched.json
  conditions/
    canny/<same-relative-image-path>
    hed/<same-relative-image-path>
    seg/<same-relative-image-path>
```

Use `scripts/prepare_xrst2i.py` to build the image/caption manifest and
`scripts/prepare_controls.py` to create condition maps. Dataset files and
generated captions are not covered by the source-code license.
