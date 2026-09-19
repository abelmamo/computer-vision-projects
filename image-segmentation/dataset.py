from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Generic image/mask segmentation dataset.

    Expects two directories with matching filenames (by stem):
        images_dir/xxx.jpg
        masks_dir/xxx.png   (single-channel, pixel value = class index;
                              for binary segmentation use 0 = background, 255 = foreground)

    Both directories are scanned for images with common extensions and paired
    by filename stem, so masks can use a different extension than the images.
    """

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        binary: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.binary = binary

        image_files = {
            p.stem: p for p in self.images_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        }
        mask_files = {
            p.stem: p for p in self.masks_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        }

        common_stems = sorted(set(image_files) & set(mask_files))
        if not common_stems:
            raise ValueError(
                f"No matching image/mask pairs found between {images_dir} and {masks_dir}"
            )

        self.pairs = [(image_files[s], mask_files[s]) for s in common_stems]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = np.array(mask, dtype=np.float32)
            if self.binary:
                mask = (mask > 127).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)  # -> (1, H, W)

        return image, mask
