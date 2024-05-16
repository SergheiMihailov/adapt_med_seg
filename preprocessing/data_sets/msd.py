"""
Dataset utilities for the MSD dataset.
defines:
- msd_image_loader
- msd_label_loader
- msd_data_download
- parse_msd: main function to parse the MSD dataset

NOTE: This script currently only supports downloading the MSD_Prostate dataset,
however adding other datasets should be straightforward.
"""
from typing import List, Dict, Tuple, Callable
import os
import json
import tarfile
import numpy as np
import gdown
from glob import glob
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ToNumpy,
    Compose
)

from util import load_callback, SPLIT_NAMES

SUPPORTED_DATASETS = ['prostate']
DATASET_URLS = {
    'prostate': '1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a'
}
DATASET_LABELS = {
    'prostate': {
        '1': 'Prostate'
    }
}
DATASET_PATHS = {
    'prostate': 'Task05_Prostate'
}
DATASET_PREFIXES = {
    'prostate': 'prostate_'
}

def msd_data_download(data_path, dataset='prostate') -> str:
    """
    Download the MSD dataset and the metadata,
    prepare the directory structure.
    OR do nothing if the data is already downloaded.
    Returns the path to the extracted data.
    """
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f'Dataset {dataset} is not supported.'\
                         'Supported datasets: {SUPPORTED_DATASETS}')

    # data_path may point to MSD_Prostate directory (which may or may not exist)
    if os.path.basename(data_path) in DATASET_PATHS.values():
        data_path = os.path.dirname(data_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    msd_data_path = os.path.join(data_path, f'MSD_{dataset}.tar')
    msd_data_dir = os.path.join(data_path, DATASET_PATHS[dataset])

    if not os.path.exists(msd_data_dir):
        # download the data
        if not os.path.exists(msd_data_path):
            print(f'Downloading {dataset} data...')
            gdown.download(f'https://drive.google.com/uc?id={DATASET_URLS[dataset]}',
                       output=msd_data_path,
                       quiet=False)
        # extract the data
        print(f'Extracting {dataset} data...')
        with tarfile.open(msd_data_path, 'r') as z:
            z.extractall(data_path)
        # remove the tar file
        if os.path.exists(msd_data_path):
            os.remove(msd_data_path)

    return msd_data_dir

msd_loader = Compose(
    [
        LoadImage(),
        EnsureChannelFirst(channel_dim="no_channel"),
        ToNumpy()
    ]
)
def msd_image_loader(image_path: str) -> np.ndarray:
    # there are two channels, the second I don't know what it is. Seems empty
    return msd_loader(image_path)[..., 0]

def msd_label_loader(label_path: str) -> np.ndarray:
    label_npy = msd_loader(label_path)
    # there are two labels, 
    # Peripheral zone (PZ) and Transition Zone (TZ)
    # we will combine them into one label
    label_npy[label_npy > 0] = 1
    return label_npy

def parse_msd(data_root: str,
               test_ratio: float,
               val_ratio: float,
               dataset: str = 'prostate') -> Tuple[Dict[str, List[Tuple[str, Callable]]],
                                                   Dict[str, List[str]],
                                                   List[str]]:
    """
        Parse the metadata for the AMOS dataset and return the data
        split as a dictionary.
    """
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f'Dataset {dataset} is not supported.'
                         'Supported datasets: {SUPPORTED_DATASETS}')

    # first make sure the data exists
    data_root = msd_data_download(data_root, dataset=dataset)

    prefix = DATASET_PREFIXES[dataset]

    # load JSON
    dataset_json_path = os.path.join(data_root, 'dataset.json')
    if not os.path.isfile(dataset_json_path):
        raise ValueError('dataset.json not found in data root')
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)

    #  create splits dictionary
    data_list = []
    modality_list = []

    images_dir = os.path.join(data_root, 'imagesTr')
    labels_dir = os.path.join(data_root, 'labelsTr')
    images_list = sorted(glob(os.path.join(images_dir, f'{prefix}*')))
    labels_list = sorted(glob(os.path.join(labels_dir, f'{prefix}*')))
    for img, lab in zip(images_list, labels_list):
        img_id = os.path.basename(img).split('.')[0][len(prefix):]
        data_list.append((str(img_id),
                          load_callback(msd_image_loader,
                                        os.path.join(images_dir, img)),
                          load_callback(msd_label_loader,
                                        os.path.join(labels_dir, lab))))
        modality_list.append('1') # MRI

    data_splits = {key: [] for key in SPLIT_NAMES}
    modality_info = {key: [] for key in SPLIT_NAMES}

    test_indices = np.random.choice(len(data_list),
                                    int(test_ratio*len(data_list)), replace=False)
    test_indices = set(test_indices)
    data_splits['test'] = [data_list[idx] for idx in test_indices]
    modality_info['test'] = [modality_list[idx] for idx in test_indices]
    rest_indices = set(range(len(data_list)))-test_indices
    val_indices = np.random.choice(list(rest_indices),
                                   int(val_ratio*len(data_list)), replace=False)
    data_splits['val'] = [data_list[idx] for idx in val_indices]
    modality_info['val'] = [modality_list[idx] for idx in val_indices]
    # remove the validation samples from the training set
    train_indices = rest_indices-val_indices
    data_splits['train'] = [data_splits['train'][idx]
                            for idx in train_indices]
    modality_info['train'] = [modality_info['train'][idx]
                              for idx in train_indices]

    # obtain list of labels and corr. classes
    classes = DATASET_LABELS[dataset]

    return data_splits, modality_info, classes
