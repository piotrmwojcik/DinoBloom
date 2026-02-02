import logging
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import torch
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger("dinov2")


class HemaStandardDataset(VisionDataset):
    def __init__(
        self,
        *,
        root: str = "",
        split: Optional[str] = None,   # e.g. "train", "pretrain", "test"
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
        extensions: Optional[List[str]] = None,  # override if needed
        recursive: bool = True,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        # Default extensions (add/remove as needed)
        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]

        base_dir = Path(root)
        if split is not None:
            base_dir = base_dir / split

        if not base_dir.exists():
            raise RuntimeError(f"Dataset directory does not exist: {base_dir}")

        exts = set(e.lower() for e in extensions)

        # Collect image file paths
        if recursive:
            files = [p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        else:
            files = [p for p in base_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]

        self.patches = [str(p) for p in sorted(files)]

        if shuffle:
            import random
            random.shuffle(self.patches)

        self.true_len = len(self.patches)
        if self.true_len == 0:
            raise RuntimeError(
                f"No image files found under: {base_dir}\n"
                f"Looked for extensions: {sorted(exts)}\n"
                f"Set recursive={recursive} if images are nested in subfolders."
            )

        logger.info(f"Loaded {self.true_len} images from {base_dir}")

    def __getitem__(self, index: int):
        # Early guard (avoids modulo by zero)
        if self.true_len == 0:
            raise RuntimeError("Dataset is empty (true_len=0). Check root/split and extensions.")

        # Try to read image; if fails, skip forward
        for _ in range(10):  # avoid infinite recursion if many corrupted files
            try:
                image, filepath = self.get_image_data(index)
                break
            except Exception as e:
                adjusted_index = index % self.true_len
                filepath = self.patches[adjusted_index]
                logger.warning(f"Cannot read image for sample {index}: {e} (path={filepath}). Skipping.")
                index += 1
        else:
            raise RuntimeError("Failed to read images after 10 attempts. Too many corrupted/missing files?")

        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, filepath

    def get_image_data(self, index: int, dimension: int = 224) -> Tuple[Image.Image, str]:
        adjusted_index = index % self.true_len
        filepath = self.patches[adjusted_index]
        patch = (
            Image.open(filepath)
            .convert("RGB")
            .resize((dimension, dimension), Image.Resampling.LANCZOS)
        )
        return patch, filepath

    def get_target(self, index: int) -> torch.Tensor:
        # labels are not used for training
        return torch.zeros((1,))

    def __len__(self) -> int:
        # Large number for infinite data sampling (keeps your original behavior)
        return 120000000
