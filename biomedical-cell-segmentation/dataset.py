from pathlib import Path

import numpy as np
import skimage.io
import torch
from skimage.color import rgb2gray
from skimage.segmentation import find_boundaries
from skimage.transform import resize as sk_resize
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def load_microscopy_image(path) -> np.ndarray:
    """
    Reads a microscopy image as a 2D float32 array scaled to roughly [0, 1].

    Color images are converted to grayscale. Intensities are normalized with
    the 1st and 99.8th percentiles rather than by dividing by 255, so 8-bit
    and 16-bit images (and dim or bright acquisitions) end up on the same
    scale. Training and inference both use this function.
    """
    image = skimage.io.imread(str(path))
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = rgb2gray(image)
    image = image.astype(np.float32)

    low, high = np.percentile(image, (1.0, 99.8))
    image = (image - low) / max(high - low, 1e-6)
    return np.clip(image, 0.0, 1.0)


def load_label_map(path) -> np.ndarray:
    """Reads an instance label map (0 = background, k = k-th nucleus)."""
    label = skimage.io.imread(str(path))
    if label.ndim == 3:
        label = label[..., 0]
    return label.astype(np.int32)


def _list_images(directory: Path):
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def find_samples(root_dir):
    """
    Returns a list of (image_path, mask_source, layout) tuples. Two layouts
    are supported:

    1. "instances" (Kaggle 2018 Data Science Bowl): one folder per sample,
       with one binary PNG per nucleus

           root_dir/<sample_id>/images/<sample_id>.png
           root_dir/<sample_id>/masks/<instance_1>.png
           root_dir/<sample_id>/masks/<instance_2>.png

    2. "labelmap" (e.g. the DSB2018 subset distributed with StarDist): one
       instance label image per sample, with matching file names

           root_dir/images/<name>.tif
           root_dir/masks/<name>.tif     (0 = background, k = k-th nucleus)
    """
    root = Path(root_dir)

    if (root / "images").is_dir() and (root / "masks").is_dir():
        samples = []
        for image_path in _list_images(root / "images"):
            mask_path = root / "masks" / image_path.name
            if mask_path.exists():
                samples.append((image_path, mask_path, "labelmap"))
        if samples:
            return samples

    samples = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if (sample_dir / "images").is_dir() and (sample_dir / "masks").is_dir():
            image_files = _list_images(sample_dir / "images")
            if image_files:
                samples.append((image_files[0], sample_dir / "masks", "instances"))

    if not samples:
        raise ValueError(
            f"No samples found under {root_dir}. Expected either "
            f"<root>/<sample_id>/images/ + masks/ subfolders, or "
            f"<root>/images/ + <root>/masks/ with matching file names."
        )
    return samples


def load_instance_label(mask_source: Path, layout: str, shape) -> np.ndarray:
    if layout == "labelmap":
        return load_label_map(mask_source)

    label = np.zeros(shape, dtype=np.int32)
    for instance_id, mask_path in enumerate(_list_images(mask_source), start=1):
        mask = skimage.io.imread(str(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        label[mask > 127] = instance_id
    return label


class NucleiDataset(Dataset):
    """
    Nuclei segmentation dataset. Each sample's instance labels are turned
    into two training targets:
      - foreground: 1 where any nucleus is present, 0 elsewhere
      - boundary:   1 on the border of each nucleus, including the border
                    between two touching nuclei

    Predicting both lets a downstream watershed step split nuclei that touch,
    which a plain foreground mask cannot do.
    """

    def __init__(self, root_dir: str, img_size: int = 256, augment: bool = False,
                 indices=None):
        self.samples = find_samples(root_dir)
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_source, layout = self.samples[idx]

        image = load_microscopy_image(image_path)
        label = load_instance_label(mask_source, layout, image.shape)

        image = sk_resize(image, (self.img_size, self.img_size),
                          preserve_range=True, anti_aliasing=True).astype(np.float32)
        label = sk_resize(label, (self.img_size, self.img_size),
                          order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)

        foreground = (label > 0).astype(np.float32)
        if label.max() > 0:
            boundary = find_boundaries(label, mode="outer").astype(np.float32)
        else:
            boundary = np.zeros_like(foreground)

        target = np.stack([foreground, boundary], axis=0)
        image = image[None]

        if self.augment:
            # Nuclei have no preferred orientation, so flips and 90-degree
            # rotations give free extra training data.
            k = np.random.randint(4)
            image = np.rot90(image, k, axes=(1, 2))
            target = np.rot90(target, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                image = image[:, :, ::-1]
                target = target[:, :, ::-1]

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float()    # (1, H, W)
        target_t = torch.from_numpy(np.ascontiguousarray(target)).float()  # (2, H, W)
        return image_t, target_t
