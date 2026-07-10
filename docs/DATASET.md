# Dataset preparation

Datasets are not distributed with this repository. Users are responsible for
obtaining them and following their licenses, access conditions, and attribution
requirements.

## Manifest schema

The dataset consumes a JSON object whose keys are image paths relative to
`images_dir` and whose values are enriched descriptions:

```json
{
  "airport/00001.jpg": "An airport with two parallel runways ..."
}
```

Control maps must repeat the same relative path under each control directory:

```text
data/images/airport/00001.jpg
data/conditions/canny/airport/00001.jpg
data/conditions/hed/airport/00001.jpg
data/conditions/seg/airport/00001.jpg
```

The training dataset returns the image in `[-1, 1]`, each control map in
`[-1, 1]`, an enriched text prompt, and a three-element active-control mask.

## xRST2I preparation

The local dataset used for validation has this source layout:

```text
xRST2I_110K/
  train/train/<id>.jpg
  train_Refined_txt/<id>.txt
```

Create a non-destructive relative-symlink view:

```bash
python scripts/prepare_xrst2i.py --root /path/to/xRST2I_110K --out data
```

Use `--count 100` for a deterministic subset, or `--mode copy` when the prepared
directory must be movable. The output stays ignored by Git.

## Control generation

Canny, HED, and EntitySeg controls are produced with:

```bash
python scripts/prepare_controls.py \
  --images-dir data/images --out-dir data/conditions \
  --types canny hed seg
```

Proxy controls are available only for smoke diagnosis and are disabled in the
release config because substituting Canny for HED or KMeans for semantic
segmentation changes the scientific experiment.

## Splits and provenance

Every reported experiment should publish a machine-readable split manifest,
the source dataset version/checksum when available, EDG backend and model
revision, preprocessing commands, and random seed. Do not commit source images
unless their license explicitly permits redistribution.
