from dataclasses import dataclass
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from SegVol.model_segvol_single import SegVolProcessor
from adapt_med_seg.utils.download_m3d_seg_dataset import download_m3d_seg_dataset
from constants import DATASETS_DIR, M3D_DATASET_DICT


@dataclass
class DataItem:
    image: torch.Tensor
    label: torch.Tensor
    zoom_out_image: torch.Tensor
    zoom_out_label: torch.Tensor


def data_item_to_device(data_item: DataItem, device: str):
    data_item["image"] = data_item["image"].to(device=device)
    data_item["label"] = data_item["label"].to(device=device)
    data_item["zoom_out_image"] = data_item["zoom_out_image"].to(device=device)
    data_item["zoom_out_label"] = data_item["zoom_out_label"].to(device=device)
    return data_item


class MedSegDataset(Dataset):
    def __init__(
        self,
        processor: SegVolProcessor,
        dataset_number: int,
        train: bool = True,
    ) -> None:
        self._processor = processor
        self._dataset_number = dataset_number
        self._name = M3D_DATASET_DICT[self.padded_dataset_number]
        self._train = train

        self.load_dataset()

    @property
    def name(self) -> str:
        return self._name

    @property
    def train(self) -> bool:
        return self._train

    def __getitem__(self, idx) -> tuple[DataItem, torch.Tensor]:
        ct_path = self._ct_paths[idx]
        gt_path = self._gt_paths[idx]

        ct_npy, gt_npy = self._processor.load_uniseg_case(ct_path, gt_path)

        data_item = self._processor.zoom_transform(ct_npy, gt_npy)

        return data_item, gt_npy

    def __len__(self):
        return len(self._ct_paths)

    @property
    def padded_dataset_number(self) -> str:
        return str(self._dataset_number).zfill(4)

    @property
    def labels(self) -> list[str]:
        return self._labels

    def load_dataset(self) -> None:
        json_path = os.path.join(
            DATASETS_DIR,
            self.padded_dataset_number,
            self.padded_dataset_number + ".json",
        )

        if not os.path.exists(json_path):
            if self._dataset_number is None:
                raise ValueError("Dataset number not provided")

            download_m3d_seg_dataset(self.padded_dataset_number)

        with open(json_path, "r", encoding="utf-8") as f:
            dataset_dict = json.load(f)

        self._labels = [
            x for _, x in dataset_dict["labels"].items() if x != "background"
        ]
        self._modality = dataset_dict["modality"]

        case_paths = dataset_dict["train"] if self.train else dataset_dict["test"]
        self._ct_paths = [
            os.path.join(DATASETS_DIR, case["image"]) for case in case_paths
        ]
        self._gt_paths = [
            os.path.join(DATASETS_DIR, case["label"]) for case in case_paths
        ]

    def get_train_val_dataloaders(
        self,
        train_val_split: float,
        batch_size: int = 1,
        seed: int = 42,
    ) -> tuple[DataLoader, DataLoader]:
        if not self.train:
            raise ValueError("This method is only for training dataset")

        total_train = len(self)
        val_size = int(total_train * train_val_split)
        train_size = total_train - val_size

        train_subset, val_subset = random_split(
            self,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        return train_loader, val_loader

    def get_test_dataloader(
        self,
        batch_size: int = 1,
    ) -> DataLoader:
        if self.train:
            raise ValueError("This method is only for test dataset")

        return DataLoader(self, batch_size=batch_size, shuffle=False)
