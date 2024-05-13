from typing import List
from dataclasses import dataclass
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from SegVol.model_segvol_single import SegVolProcessor
from adapt_med_seg.utils.download_m3d_seg_dataset import download_m3d_seg_dataset
from constants import DATASETS_DIR


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
        dataset_path: int,
        modalities: List[str],
        train: bool = True
    ) -> None:
        self._processor = processor
        self._dataset_path = dataset_path
        self._modalities = modalities
        self._train = train

        self.load_dataset()

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, idx) -> tuple[DataItem, torch.Tensor, torch.Tensor]:
        ct_path = self._ct_paths[idx]
        gt_path = self._gt_paths[idx]
        modality = self._case_modality[idx]

        ct_npy, gt_npy = self._processor.load_uniseg_case(ct_path, gt_path)
        modality_tensor = torch.tensor([modality])

        if self.train and idx not in getattr(self, "_test_indices", []):
            data_item = self._processor.train_transform(ct_npy, gt_npy)
        else:
            data_item = self._processor.zoom_transform(ct_npy, gt_npy)

        return data_item, gt_npy, modality_tensor

    def __len__(self):
        return len(self._ct_paths)

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    @property
    def dataset_number(self) -> str:
        return self._dataset_number

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def train(self) -> bool:
        return self._train

    @property
    def modality_id2name(self) -> dict[int, str]:
        return self._modality_id2name

    @property
    def modality_name2id(self) -> dict[str, int]:
        return self._modality_name2id

    @property
    def modalities(self) -> str:
        return self._modalities

    def load_dataset(self) -> None:
        """
            Loads the dataset from the dataset.json file
            sets static attributes:
            - name
            - dataset_number
            - modality mapping (e.g. {"CT": 0, "MR": 1, "PET": 2})
            - label mapping (e.g. {"liver": 1, "tumor": 2, "background": 0})
            loads the paths of the CT and GT files and the corresponding modality
            for each case in the dataset and
            creates a Subset for each split (train, val) or (test), 
            depending on the train attribute.
        """
        json_path = os.path.join(self._dataset_path, "dataset.json")
        
        if not os.path.exists(json_path):
            raise ValueError("Dataset JSON not found")

        with open(json_path, "r", encoding="utf-8") as f:
            dataset_dict = json.load(f)

        self._name = dataset_dict["name"]
        self._dataset_number = dataset_dict["description"]

        self._modality_id2name = dataset_dict["modality"]
        self._modality_name2id = {  #  we will likely need this
            v: k for k, v in self._modality_id2name.items()
        }
        mod_ids = set([self._modality_name2id[mod]
                      for mod in self._modalities])

        self._labels = [
            x for _, x in dataset_dict["labels"].items() if x != "background"
        ]
        splits = ["train", "val"] if self.train else ["test"]
        self._ct_paths = []
        self._gt_paths = []
        self._case_modality = []
        data_idxs = {}
        base = 0
        for split in splits:
            data_idxs[split] = []
            case_paths = dataset_dict[split]
            idx = 0
            for case_ in case_paths:
                if case_["modality"] not in mod_ids:
                    continue
                self._ct_paths.append(
                    os.path.join(self.dataset_path, case_["image"]))
                self._gt_paths.append(
                    os.path.join(self.dataset_path, case_["label"]))
                self._case_modality.append(int(case_["modality"]))
                data_idxs[split].append(base + idx)
                idx += 1
            base += idx
        if self.train:
            # create subsets for each split
            self._tr_val_splits = {split: Subset(self, data_idxs[split])
                                   for split in splits}

    def get_train_val_dataloaders(
        self,
        batch_size: int = 1
    ) -> tuple[DataLoader, DataLoader]:
        if not self.train:
            raise ValueError("This method is only for training dataset")

        train_subset = self._tr_val_splits["train"]
        val_subset = self._tr_val_splits["val"]

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        # Hack to get the test indices for __getitem__
        self._test_indices = val_subset.indices
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
