from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from src.config import (
    BATCH_SIZE,
    IMAGE_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    MASK_DIR,
    RANDOM_SEED,
    VAL_BATCH,
)


def build_dataframes(
    image_dir: Path = IMAGE_DIR,
    mask_dir: Path = MASK_DIR,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan image_dir and mask_dir, split into train, val, and test dataframes."""
    image_files = sorted(image_dir.glob("*.jpg"))
    records = []
    for img_path in image_files:
        mask_path = mask_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            records.append({
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "file_name": img_path.name,
            })

    df = pd.DataFrame(records)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=random_seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_seed)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return df, train_df, val_df, test_df


class BinarySegmentationDataset(Dataset):
    """Loads image-mask pairs, normalizes images, and binarizes target masks."""

    def __init__(self, df: pd.DataFrame, img_size: tuple[int, int] = IMG_SIZE):
        self.df = df
        self.img_size = img_size

        self.img_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.mask_resize = T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image_tensor = self.img_transform(image)

        mask = Image.open(row["mask_path"]).convert("L")
        mask = self.mask_resize(mask)
        mask_np = np.array(mask, dtype=np.uint8)

        threshold = 0 if mask_np.max() <= 1 else 128
        mask_tensor = torch.from_numpy(mask_np > threshold).long()

        return image_tensor, mask_tensor


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    img_size: tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    val_batch: int = VAL_BATCH,
) -> tuple:
    """Build train, val, and test datasets and DataLoader instances."""
    train_ds = BinarySegmentationDataset(train_df, img_size=img_size)
    val_ds = BinarySegmentationDataset(val_df, img_size=img_size)
    test_ds = BinarySegmentationDataset(test_df, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=val_batch, shuffle=False, num_workers=0)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
