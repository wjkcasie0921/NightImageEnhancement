from pathlib import Path
import random
from typing import List, Tuple

from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


class LowLightPairDataset(Dataset):
    """
    Paired low-light dataset for the LOL-style directory layout.

    Supported structure:
      root/train/low
      root/train/normal
      root/eval15/low
      root/eval15/normal
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
    ):
        super().__init__()
        if split not in {"train", "eval15"}:
            raise ValueError(f"Unsupported split: {split}. Expected 'train' or 'eval15'.")

        self.root = Path(root)
        self.split = split
        self.augment = augment and split == "train"

        self.low_dir = self.root / split / "low"
        self.normal_dir = self.root / split / "normal"

        if not self.low_dir.exists():
            raise FileNotFoundError(
                f"Low-light directory not found: {self.low_dir}. Expected structure: {self.root}/{split}/low"
            )
        if not self.normal_dir.exists():
            raise FileNotFoundError(
                f"Normal-light directory not found: {self.normal_dir}. Expected structure: {self.root}/{split}/normal"
            )

        self.low_images = _list_images(self.low_dir)
        self.normal_images = _list_images(self.normal_dir)

        if len(self.low_images) == 0:
            raise RuntimeError(f"No low-light images found in {self.low_dir}.")
        if len(self.normal_images) == 0:
            raise RuntimeError(f"No normal-light images found in {self.normal_dir}.")

        normal_lookup = {p.stem: p for p in self.normal_images}
        paired_low: List[Path] = []
        paired_normal: List[Path] = []
        for low_path in self.low_images:
            normal_path = normal_lookup.get(low_path.stem)
            if normal_path is None:
                continue
            paired_low.append(low_path)
            paired_normal.append(normal_path)

        if len(paired_low) == 0:
            raise RuntimeError(
                f"No valid low/normal pairs found in {self.low_dir} and {self.normal_dir}. "
                "Please ensure filenames share the same stem."
            )

        skipped = len(self.low_images) - len(paired_low)
        if skipped > 0:
            print(
                f"[{self.split}] Warning: {skipped} low-light images have no normal match and were skipped."
            )

        self.low_images = paired_low
        self.normal_images = paired_normal

        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.normal_images)

    @staticmethod
    def _synthesize_low_image(img: Image.Image) -> Image.Image:
        """
        Simulate low-light input from normal-light image.
        This fallback is useful when only normal images are available.
        """
        brightness_factor = random.uniform(0.2, 0.6)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        contrast_factor = random.uniform(0.6, 0.9)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        return img

    @staticmethod
    def _paired_augment(low: torch.Tensor, normal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply same spatial augmentations to preserve pair alignment."""
        if random.random() < 0.5:
            low = torch.flip(low, dims=[2])
            normal = torch.flip(normal, dims=[2])
        if random.random() < 0.5:
            low = torch.flip(low, dims=[1])
            normal = torch.flip(normal, dims=[1])
        return low, normal

    def __getitem__(self, idx: int):
        normal_img = Image.open(self.normal_images[idx]).convert("RGB")
        low_img = Image.open(self.low_images[idx]).convert("RGB")

        low = self.to_tensor(low_img)
        normal = self.to_tensor(normal_img)

        if self.augment:
            low, normal = self._paired_augment(low, normal)

        return low, normal