import torch
from pathlib import Path
from torch.utils.data import Dataset
from os import listdir
from PIL import Image
from typing import Tuple, List


class FaceMaskDataset(Dataset):
    def __init__(self, dataset_dir: Path, transform=None):
        self.img_labels: List[Tuple[str, int]] = self.get_files_dataset(dataset_dir)
        self.dataset_dir = dataset_dir
        self.transform = transform

    @staticmethod
    def get_files_dataset(dataset_dir: Path):
        with_mask_files = [f"WithMask/{i}" for i in listdir(dataset_dir / "WithMask")]
        without_mask_files = [
            f"WithoutMask/{i}" for i in listdir(dataset_dir / "WithoutMask")
        ]
        return list(
            zip(
                with_mask_files + without_mask_files,
                [1] * len(with_mask_files) + [0] * len(without_mask_files),
            )
        )

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.dataset_dir / self.img_labels[idx][0]
        image = Image.open(str(img_path.resolve()))
        label = self.img_labels[idx][1]

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), label
