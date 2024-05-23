from typing import List, Tuple, Dict, Any, NamedTuple
from dataclasses import dataclass
from glob import glob
import numpy as np
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
        train: bool = True,
    ) -> None:
        self._processor = processor
        self._dataset_path = dataset_path
        self._modalities = modalities
        self._train = train

        self.load_dataset()

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, idx) -> tuple[DataItem, torch.Tensor, str, str]:
        ct_path = self._ct_paths[idx]
        gt_path = self._gt_paths[idx]
        modality = str(self._case_modality[idx])
        label_map = self._case_label[idx]
        # global is what we return as the label (e.g. pancreas)
        global_label = self._labels[label_map["global_idx"]]
        # local label is what we use to index the ground truth
        local_label = int(label_map["local_idx"])

        ct_npy, gt_npy = self._processor.load_uniseg_case(ct_path, gt_path)

        # retrieve the mask, but keep the channel dimension
        gt_npy = np.expand_dims(gt_npy[local_label - 1, ...], axis=0)

        ct_npy = ct_npy.astype("float32")
        gt_npy = gt_npy.astype("int64")

        if self._train:
            data_item = self._processor.train_transform(ct_npy, gt_npy)
        else:
            data_item = self._processor.zoom_transform(ct_npy, gt_npy)

        return data_item, gt_npy, modality, global_label

    def __len__(self):
        return len(self._ct_paths)

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
            raise FileNotFoundError(
                f"Dataset path {self.dataset_path} does not exist in directory {os.getcwd()}"
            )
        if os.path.exists(os.path.join(self.dataset_path, "dataset.json")):
            dataset_paths = [(self.dataset_path, "dataset.json")]
        else:
            dataset_paths = [
                (dirpath, path)
                # use os.walk to get all subdirectories
                for dirpath, _, files in os.walk(self.dataset_path)
                for path in files
                if path.endswith(".json")
            ]
        if dataset_paths == []:
            raise FileNotFoundError(
                "No dataset.json found in the dataset folder or any of its subdirectories"
            )

        # figure out the global label mapping and override the
        # dataset-specific labels with the global label ids
        # creates:
        # - self._labels: list of global label names
        # - self._labels_inv: dict of global label names to ids
        # - self._data_dict: dict of json data for each dataset
        self._load_and_gather_labels(dataset_paths)

        self._modality_id2name = {}
        self._modality_name2id = {}

        self._ct_paths = []
        self._gt_paths = []
        self._case_modality = []
        self._case_label = []
        # M3D-Seg does not specify a validation split so we re-use the test split.
        # not ideal but oh well...
        splits = (
            [("training", "train"), ("validation", "test")]
            if self.train
            else [("test", "test")]
        )
        self._data_idxs = {split[0]: [] for split in splits}

        dataset_names = []
        dataset_numbers = []
        for json_path in self._data_dict.keys():
            dataset_name, dataset_number = self._load_single_dataset(json_path, splits)
            dataset_names.append(dataset_name)
            dataset_numbers.append(dataset_number)

        self._name = (
            dataset_names[0]
            if len(dataset_names) == 1
            else f"{'/'.join(dataset_names)}"
        )
        self._dataset_number = (
            dataset_numbers[0]
            if len(dataset_numbers) == 1
            else f"{'/'.join(dataset_numbers)}"
        )

        # shuffle the data indices, otherwise they will be sorted by dataset
        for split in splits:
            self._data_idxs[split[0]] = torch.tensor(self._data_idxs[split[0]])[
                torch.randperm(len(self._data_idxs[split[0]]))
            ]

        if self.train:
            # create subsets for each split
            self._tr_val_splits = {
                split[0]: Subset(self, self._data_idxs[split[0]]) for split in splits
            }
        else:
            self._test_splits = {
                split[0]: Subset(self, self._data_idxs[split[0]]) for split in splits
            }

    def _load_and_gather_labels(self, dataset_paths: list[Tuple[str, str]]):
        label_set = set()
        self._data_dict = {}
        for dataset_path, json_name in dataset_paths:
            json_path = os.path.join(dataset_path, json_name)
            # load json data
            with open(json_path, "r", encoding="utf-8") as f:
                self._data_dict[json_path] = json.load(f)
            # gather labels
            label_set = label_set.union(
                set(
                    [
                        x
                        for _, x in self._data_dict[json_path]["labels"].items()
                        if x != "background"
                    ]
                )
            )
        # create a label mapping global_index -> (liver, pancreas, ...)
        self._labels = {str(idx): label for idx, label in enumerate(label_set)}
        # name to id map (liver, pancreas, ...) -> global_index
        self._labels_inv = {v: k for k, v in self._labels.items()}
        # map dataset-specific labels to the global label mapping
        for json_path, data in self._data_dict.items():
            # override the labels field with <global_label_id>: <local_label_id>
            data["labels"] = {
                self._labels_inv[label]: idx
                for idx, label in data[
                    "labels"
                ].items()  # excluding any labels not in the global label mapping
                if label in self._labels_inv
            }

    def _load_single_dataset(
        self, data_json: str, splits: List[str | Tuple[str, str]]
    ) -> None:
        data_dict: Dict[str, Any] = self._data_dict[data_json]
        data_path: str = os.path.dirname(data_json)

        name = data_dict["name"]
        dataset_number = data_dict["description"]

        self._modality_id2name.update(
            data_dict.get("modality", {"0": "CT", "1": "MRI"})
        )
        self._modality_name2id.update(
            {  #  we will likely need this
                v: k for k, v in self._modality_id2name.items()
            }
        )
        mod_ids = set(
            [
                self._modality_name2id[mod]
                for mod in self._modalities
                if mod in self._modality_name2id
            ]
        )

        # each datapoint will correspond to K dataset indices,
        # one for each local label k
        local_label_map = [
            {"global_idx": global_idx, "local_idx": local_idx}
            for global_idx, local_idx in data_dict["labels"].items()
        ]

        base = len(self._ct_paths)
        for split, m3d_split in splits:
            case_paths = data_dict.get(split, data_dict.get(m3d_split, None))
            idx = 0
            for case_ in case_paths:
                # default to '0': 'CT', if not specified
                if case_.get("modality", "0") not in mod_ids:
                    continue
                ct_path = os.path.join(data_path, case_["image"])
                gt_path = os.path.join(data_path, case_["label"])
                if not os.path.isfile(ct_path):
                    # the M3D seg way
                    ct_path = os.path.join(os.path.dirname(data_path), case_["image"])
                if not os.path.isfile(gt_path):
                    # the M3D seg way
                    gt_path = os.path.join(os.path.dirname(data_path), case_["label"])
                # append a new item for each seprate label in the case
                for llm in local_label_map:
                    self._ct_paths.append(ct_path)
                    self._gt_paths.append(gt_path)
                    self._case_modality.append(int(case_.get("modality", "0")))
                    self._case_label.append(llm)
                    self._data_idxs[split].append(base + idx)
                    idx += 1
            base += idx

        return name, dataset_number

    def get_train_val_dataloaders(
        self, batch_size: int = 1
    ) -> tuple[DataLoader, DataLoader]:
        if not self.train:
            raise ValueError("This method is only for training dataset")

        train_subset = self._tr_val_splits["training"]
        val_subset = self._tr_val_splits["validation"]

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
        test_dataloader = DataLoader(
            self._test_splits["test"], batch_size=batch_size, shuffle=False
        )
        return test_dataloader

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
    def labels_inv(self) -> dict[str, int]:
        return self._labels_inv

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

    @property
    def data_dict(self) -> dict[str, Any]:
        return self._data_dict
