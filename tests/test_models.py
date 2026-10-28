"""
Fast checks that run on CPU in a few seconds, with synthetic data only.
They cover model output shapes, target derivation and the instance
matching used for evaluation, so a refactor that breaks any of them fails CI.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(project: str, name: str):
    """Imports <project>/<name>.py with that project folder on sys.path."""
    project_dir = str(ROOT / project)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    for cached in ("dataset", "unet", "model", "segment", "train", "evaluate"):
        sys.modules.pop(cached, None)
    spec = importlib.util.spec_from_file_location(f"{project}_{name}", ROOT / project / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.remove(project_dir)
    return module


def test_classifier_output_shape():
    model_mod = load_module("image-classifier", "model")
    model = model_mod.SimpleCNN(num_classes=10).eval()
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)


@pytest.mark.parametrize("size", [64, 100])
def test_binary_unet_preserves_size(size):
    unet_mod = load_module("image-segmentation", "unet")
    model = unet_mod.UNet(in_channels=3, num_classes=1, base_channels=16).eval()
    out = model(torch.randn(1, 3, size, size))
    assert out.shape[-2:] == (size, size)


def test_nuclei_unet_has_two_output_channels():
    unet_mod = load_module("biomedical-cell-segmentation", "unet")
    model = unet_mod.NucleiUNet(in_channels=1).eval()
    out = model(torch.randn(2, 1, 64, 64))
    assert out.shape == (2, 2, 64, 64)


def test_nuclei_targets_from_label_map(tmp_path):
    import skimage.io

    dataset_mod = load_module("biomedical-cell-segmentation", "dataset")
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()

    image = np.zeros((40, 40), dtype=np.uint8)
    label = np.zeros((40, 40), dtype=np.uint16)
    label[5:15, 5:15] = 1
    label[15:25, 5:15] = 2  # touches nucleus 1
    image[label > 0] = 200
    skimage.io.imsave(tmp_path / "images" / "a.tif", image, check_contrast=False)
    skimage.io.imsave(tmp_path / "masks" / "a.tif", label, check_contrast=False)

    ds = dataset_mod.NucleiDataset(str(tmp_path), img_size=40)
    x, y = ds[0]
    assert x.shape == (1, 40, 40) and y.shape == (2, 40, 40)
    assert y[0].sum() == 200  # two 10x10 nuclei
    # The border between the two touching nuclei must be marked as boundary.
    assert y[1, 14:16, 8].sum() > 0


def test_instance_matching_counts():
    eval_mod = load_module("biomedical-cell-segmentation", "evaluate")
    gt = np.zeros((10, 10), dtype=int)
    gt[0:4, 0:4] = 1
    gt[6:9, 6:9] = 2
    pred = np.zeros((10, 10), dtype=int)
    pred[0:4, 0:3] = 1   # IoU 0.75 with gt 1 -> true positive
    pred[6:9, 6:7] = 2   # IoU 0.33 with gt 2 -> false positive, gt 2 missed
    pred[0:2, 8:10] = 3  # no overlap -> false positive
    assert eval_mod.match_instances(gt, pred) == (1, 2, 1)


def test_watershed_splits_touching_nuclei():
    segment_mod = load_module("biomedical-cell-segmentation", "segment")
    foreground = np.zeros((20, 30))
    foreground[5:15, 3:27] = 1.0      # one blob made of two nuclei
    boundary = np.zeros((20, 30))
    boundary[5:15, 14:16] = 1.0       # predicted border between them
    labels = segment_mod.instances_from_maps(foreground, boundary)
    assert labels.max() == 2
