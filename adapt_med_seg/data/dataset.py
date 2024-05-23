from typing import List, Tuple
from dataclasses import dataclass
from glob import glob
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from SegVol.model_segvol_single import SegVolProcessor


@dataclass
class DataItem:
    image: torch.Tensor
    label: torch.Tensor
    zoom_out_image: torch.Tensor
    zoom_out_label: torch.Tensor


def data_item_to_device(data_item: DataItem, device: str):
    data_item["image"] = data_item["image"].to(device=device)
    data_item["label"] = data_item["label"].to(device=device)

    if "zoom_out_image" in data_item:
        data_item["zoom_out_image"] = data_item["zoom_out_image"].to(device=device)

    if "zoom_out_label" in data_item:
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

    def __getitem__(self, idx) -> tuple[DataItem, torch.Tensor, str]:
        ct_path = self._ct_paths[idx]
        gt_path = self._gt_paths[idx]
        modality = str(self._case_modality[idx])

        ct_npy, gt_npy = self._processor.load_uniseg_case(ct_path, gt_path)

        ct_npy = ct_npy.astype("float32")
        gt_npy = gt_npy.astype("int64")

        if self.train and idx not in getattr(self, "_test_indices", []):
            data_item = self._processor.train_transform(ct_npy, gt_npy)
        else:
            data_item = self._processor.zoom_transform(ct_npy, gt_npy)

        return data_item, gt_npy, modality

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
            Loads the dataset from the dataset.json file.
            If no dataset.json is found in the root of the dataset folder,
            it recursively searches for multiple dataset.json files in the
            subdirectories of the dataset folder. 
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

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist in directory {os.getcwd()}")
        if os.path.exists(os.path.join(self.dataset_path, "dataset.json")):
            dataset_paths = [(self.dataset_path, 'dataset.json')]
        else:
            dataset_paths = [
                (dirpath, path)
                # use os.walk to get all subdirectories
                for dirpath, _, files in os.walk(self.dataset_path)
                for path in files if path.endswith(".json")
            ]
        if dataset_paths == []:
            raise FileNotFoundError("No dataset.json found in the dataset folder or any of its subdirectories")

        self._modality_id2name = {}
        self._modality_name2id = {}
        self._labels = []
        self._ct_paths = []
        self._gt_paths = []
        self._case_modality = []
        # M3D-Seg does not specify a validation split so we re-use the test split.
        # not ideal but oh well...
        splits = [("training", "train"), ("validation", "test")]\
            if self.train else ["test"]
        self.data_idxs = {split[0]: [] for split in splits}

        dataset_names = []
        dataset_numbers = []
        for dataset_path, json_name in dataset_paths:
            dataset_name, dataset_number = self._load_single_dataset(dataset_path, json_name, splits)
            dataset_names.append(dataset_name)
            dataset_numbers.append(dataset_number)

        self._name = dataset_names[0] if len(dataset_names) == 1 else \
            f"{'/'.join(dataset_names)}"
        self._dataset_number = dataset_numbers[0] if len(dataset_numbers) == 1 else \
            f"{'/'.join(dataset_numbers)}"

        # shuffle the data indices, otherwise they will be sorted by dataset
        for split in splits:
            self.data_idxs[split[0]] = torch.tensor(
                self.data_idxs[split[0]])[torch.randperm(len(self.data_idxs[split[0]]))]

        if self.train:
            # create subsets for each split
            self._tr_val_splits = {split: Subset(self, self.data_idxs[split[0]])
                                   for split in splits}


    def _load_single_dataset(self, dataset_path: str, json_name: str, splits: List[str|Tuple[str,str]]) -> None:
        json_path = os.path.join(dataset_path, json_name)
        with open(json_path, "r", encoding="utf-8") as f:
            dataset_dict = json.load(f)

        name = dataset_dict["name"]
        dataset_number = dataset_dict["description"]

        self._modality_id2name.update(dataset_dict.get("modality", {'0': 'CT', '1': 'MRI'}))
        self._modality_name2id.update({  #  we will likely need this
            v: k for k, v in self._modality_id2name.items()
        })
        mod_ids = set([self._modality_name2id[mod]
                      for mod in self._modalities if mod in self._modality_name2id])

        # concatenate labels
        self._labels = list(
            set(self._labels).union(set([
            x for _, x in dataset_dict["labels"].items() if x != "background"
        ])))

        base = len(self._ct_paths)
        for split, m3d_split in splits:
            case_paths = dataset_dict.get(split, dataset_dict.get(m3d_split, None))
            idx = 0
            for case_ in case_paths:
                # default to '0': 'CT', if not specified
                if case_.get("modality", '0') not in mod_ids:
                    continue
                ct_path = os.path.join(dataset_path, case_["image"])
                gt_path = os.path.join(dataset_path, case_["label"])
                if not os.path.isfile(ct_path):
                    # the M3D seg way
                    ct_path = os.path.join(os.path.dirname(dataset_path), case_["image"])
                if not os.path.isfile(gt_path):
                    # the M3D seg way
                    gt_path = os.path.join(os.path.dirname(dataset_path), case_["label"])
                self._ct_paths.append(ct_path)
                self._gt_paths.append(gt_path)
                self._case_modality.append(int(case_.get("modality", '0')))
                self.data_idxs[split].append(base + idx)
                idx += 1
            base += idx

        return name, dataset_number

    def get_train_val_dataloaders(
        self,
        batch_size: int = 1
    ) -> tuple[DataLoader, DataLoader]:
        if not self.train:
            raise ValueError("This method is only for training dataset")

        train_subset = self._tr_val_splits["training"]
        val_subset = self._tr_val_splits["validation"]

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
