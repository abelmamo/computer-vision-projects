from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import find_boundaries
from skimage.transform import resize as sk_resize
from torch.utils.data import Dataset


class NucleiDataset(Dataset):
    """
    Loads data in the Kaggle "2018 Data Science Bowl" nuclei-segmentation
    layout, which is a common format for public microscopy segmentation
    datasets:

        root_dir/
          <sample_id>/
            images/<sample_id>.png
            masks/<instance_1>.png   (one binary PNG per individual cell/nucleus)
            masks/<instance_2>.png
            ...

    Each sample's per-instance masks are merged into a single instance-label
    map, from which we derive two training targets:
      - foreground: 1 where any cell is present, 0 elsewhere
      - boundary:   1 on the border between touching instances

    Predicting both lets a downstream watershed step split cells that are
    touching each other, which a plain foreground mask alone cannot do.
    """

    def __init__(self, root_dir: str, img_size: int = 256):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.sample_dirs = sorted(
            p for p in self.root_dir.iterdir()
            if p.is_dir() and (p / "images").exists() and (p / "masks").exists()
        )

        if not self.sample_dirs:
            raise ValueError(
                f"No samples found under {root_dir}. Expected "
                f"<root_dir>/<sample_id>/images/ and .../masks/ subfolders."
            )

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def _load_image(self, sample_dir: Path) -> np.ndarray:
        image_files = list((sample_dir / "images").glob("*"))
        image = Image.open(image_files[0]).convert("L")  # grayscale, typical for microscopy
        image = np.array(image, dtype=np.float32) / 255.0
        return image

    def _load_instance_label(self, sample_dir: Path, shape) -> np.ndarray:
        label = np.zeros(shape, dtype=np.int32)
        mask_files = sorted((sample_dir / "masks").glob("*"))

        for instance_id, mask_path in enumerate(mask_files, start=1):
            mask = np.array(Image.open(mask_path).convert("L"))
            label[mask > 127] = instance_id

        return label

    def __getitem__(self, idx: int):
        sample_dir = self.sample_dirs[idx]

        image = self._load_image(sample_dir)
        label = self._load_instance_label(sample_dir, image.shape)

        image = sk_resize(image, (self.img_size, self.img_size),
                           preserve_range=True, anti_aliasing=True)
        label = sk_resize(label, (self.img_size, self.img_size),
                           order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)

        foreground = (label > 0).astype(np.float32)

        if label.max() > 0:
            boundary = find_boundaries(label, mode="outer").astype(np.float32)
        else:
            boundary = np.zeros_like(foreground)

        image_t = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        target_t = torch.from_numpy(np.stack([foreground, boundary], axis=0)).float()  # (2, H, W)

        return image_t, target_t
