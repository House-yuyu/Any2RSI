"""
src/data/rst2i_dataset.py

Remote-sensing T2I dataset for Any2RSI. Returns a sample dict with the keys the
Any2RSIControlLDM expects:
    jpg          : target image, HWC float in [-1,1]   (LDM first_stage_key)
    txt/caption  : enriched caption string             (cond_stage_key)
    hint_<ctrl>  : control map, HWC float in [-1,1]
    active       : [n] mask of present controls

Captions come from the EDG json produced by scripts/edg_generate.py.
A random subset of controls is kept per sample; dropped ones are zero maps so the
model learns to handle arbitrary control combinations (as in AnyControl).
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.data.controls import CONTROL_FNS


class RST2IDataset(Dataset):
    def __init__(self, images_dir, enriched_json, control_types=("canny", "hed", "seg"),
                 image_size=512, min_controls=1, max_controls=3,
                 hed_annotator=None, seg_annotator=None, controls_dir=None,
                 allow_proxy_controls=False, drop_text_prob=0.05,
                 drop_all_prob=0.05, split=None):
        self.images_dir = images_dir
        with open(enriched_json, "r", encoding="utf-8") as handle:
            self.captions = json.load(handle)
        names = [n for n in self.captions
                 if os.path.exists(os.path.join(images_dir, n))]
        if split is not None:
            if isinstance(split, str):
                with open(split, "r", encoding="utf-8") as handle:
                    split_names = {line.strip() for line in handle if line.strip()}
            else:
                split_names = set(split)
            names = [name for name in names if name in split_names]
        self.names = sorted(names)
        if not self.names:
            raise ValueError("no captioned images were found for RST2IDataset")
        self.control_types = list(control_types)
        unknown_controls = set(self.control_types) - set(CONTROL_FNS)
        if unknown_controls:
            raise ValueError(f"unknown control types: {sorted(unknown_controls)}")
        self.image_size = image_size
        self.min_controls = min_controls
        self.max_controls = max_controls
        self.annotators = {"hed": hed_annotator, "seg": seg_annotator}
        self.controls_dir = controls_dir
        self.allow_proxy_controls = allow_proxy_controls
        self.drop_text_prob = float(drop_text_prob)
        self.drop_all_prob = float(drop_all_prob)
        if (self.drop_text_prob < 0 or self.drop_all_prob < 0 or
                self.drop_text_prob + self.drop_all_prob > 1):
            raise ValueError("drop probabilities must be non-negative and sum to <= 1")
        if (self.min_controls < 1 or self.max_controls < self.min_controls or
                self.min_controls > len(self.control_types)):
            raise ValueError(
                "require 1 <= min_controls <= max_controls and min_controls "
                "<= len(control_types)"
            )

    def __len__(self):
        return len(self.names)

    def _load(self, name):
        img = Image.open(os.path.join(self.images_dir, name)).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        return np.asarray(img)

    def _control_path(self, control_type, name):
        if self.controls_dir is None:
            return None
        if hasattr(self.controls_dir, "get"):
            root = self.controls_dir.get(control_type)
            return os.path.join(root, name) if root else None
        return os.path.join(self.controls_dir, control_type, name)

    def _load_control(self, control_type, name, image):
        path = self._control_path(control_type, name)
        if path and os.path.exists(path):
            control = Image.open(path).convert("RGB")
            control = control.resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )
            return np.asarray(control)

        annotator = self.annotators.get(control_type)
        if control_type in ("hed", "seg") and annotator is None:
            if not self.allow_proxy_controls:
                raise FileNotFoundError(
                    f"missing precomputed {control_type!r} map for {name!r}. "
                    "Run scripts/prepare_controls.py or explicitly set "
                    "allow_proxy_controls=true for a non-faithful smoke test."
                )
        fn = CONTROL_FNS[control_type]
        return (fn(image, annotator)
                if control_type in ("hed", "seg") else fn(image))

    @staticmethod
    def _norm(arr):
        return (arr.astype(np.float32) / 127.5) - 1.0   # HWC in [-1,1]

    def __getitem__(self, idx):
        name = self.names[idx]
        img_np = self._load(name)
        caption = self.captions[name]
        if isinstance(caption, list):
            caption = random.choice(caption)
        if not isinstance(caption, str):
            raise TypeError(f"caption for {name!r} must be a string or string list")

        k = random.randint(self.min_controls,
                            min(self.max_controls, len(self.control_types)))
        chosen = set(random.sample(self.control_types, k))

        draw = random.random()
        drop_all = draw < self.drop_all_prob
        drop_text = drop_all or draw < self.drop_all_prob + self.drop_text_prob
        if drop_all:
            chosen.clear()

        sample = {
            "jpg": self._norm(img_np),
            "txt": "" if drop_text else caption,
            "caption": caption,
        }
        active = []
        for ct in self.control_types:
            if ct in chosen:
                cmap = self._load_control(ct, name, img_np)
                sample[f"hint_{ct}"] = self._norm(cmap)
                active.append(1.0)
            else:
                sample[f"hint_{ct}"] = np.zeros_like(img_np, dtype=np.float32)
                active.append(0.0)
        sample["active"] = np.asarray(active, dtype=np.float32)
        return sample
