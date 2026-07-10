import json

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from src.data.rst2i_dataset import RST2IDataset


def test_precomputed_controls_and_active_mask(tmp_path):
    image_root = tmp_path / "images"
    control_root = tmp_path / "controls"
    image_root.mkdir()
    array = np.full((16, 16, 3), 127, dtype=np.uint8)
    Image.fromarray(array).save(image_root / "a.png")
    for control_type in ("canny", "hed", "seg"):
        (control_root / control_type).mkdir(parents=True)
        Image.fromarray(array).save(control_root / control_type / "a.png")
    captions = tmp_path / "captions.json"
    captions.write_text(json.dumps({"a.png": "an airport"}), encoding="utf-8")

    dataset = RST2IDataset(
        str(image_root), str(captions), controls_dir=str(control_root),
        image_size=16, min_controls=3, max_controls=3,
        drop_text_prob=0, drop_all_prob=0,
    )
    sample = dataset[0]
    assert sample["active"].tolist() == [1.0, 1.0, 1.0]
    assert sample["txt"] == "an airport"
    assert sample["jpg"].shape == (16, 16, 3)
