"""
Dataset utilities for the PROMISE12 dataset.
defines:
- promise12_image_loader
- promise12_label_loader
- promise12_data_download
- parse_promise12
"""

import os
import glob
import zipfile as zf
import requests
import numpy as np
from typing import List, Dict, Tuple, Callable
import SimpleITK as sitk
from util import load_callback, SPLIT_NAMES, two_way_split

def promise12_data_download(data_root: str) -> str:
    """
        Download the PROMISE12 dataset and the metadata,
        perpare the directory structure.
        OR do nothing if the data is already downloaded.

        Returns the path to the extracted data.
    """
    promise12_train_url = 'https://zenodo.org/records/8026660/files/training_data.zip'
    promise12_test_url = 'https://zenodo.org/records/8026660/files/test_data.zip'
    promise12_train_path = os.path.join(data_root, 'training_data.zip')
    promise12_test_path = os.path.join(data_root, 'test_data.zip')
    promise12_train_dir = os.path.join(data_root, 'training_data')
    promise12_test_dir = os.path.join(data_root, 'test_data')
    # data_root is where we extract the data
    os.makedirs(data_root, exist_ok=True)

    # download and extract the data
    if not os.path.exists(promise12_train_dir):
        if not os.path.exists(promise12_train_path):
            print('Downloading PROMISE12 training data...')
            r = requests.get(promise12_train_url, allow_redirects=True)
            with open(promise12_train_path, 'wb') as f:
                f.write(r.content)
        # extract the data
        print('Extracting PROMISE12 training data...')
        with zf.ZipFile(promise12_train_path, 'r') as z:
            z.extractall(data_root)
        # remove the zip file
        if os.path.exists(promise12_train_path):
            os.remove(promise12_train_path)
    if not os.path.exists(promise12_test_dir):
        if not os.path.exists(promise12_test_path):
            print('Downloading PROMISE12 test data...')
            r = requests.get(promise12_test_url, allow_redirects=True)
            with open(promise12_test_path, 'wb') as f:
                f.write(r.content)
        # extract the data
        print('Extracting PROMISE12 test data...')
        with zf.ZipFile(promise12_test_path, 'r') as z:
            z.extractall(data_root)
        # remove the zip file
        if os.path.exists(promise12_test_path):
            os.remove(promise12_test_path)

    return data_root

promise_labels = {
    '1': 'prostate'
}

def promise12_image_loader(image_path):
    """
    Load the image from the given path.
    """
    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 2, 3, 1))
    return image

promise12_label_loader = promise12_image_loader

def parse_promise12(data_root: str,
                    val_ratio: float) -> Tuple[Dict[str, List[Tuple[int, Callable]]],
                                        Dict[str, List[str]],
                                        List[str]]:
    """
    Parse the PROMISE12 dataset.
    Returns the data splits, modality info and classes.
    """
    # make sure the data exists
    data_root = promise12_data_download(data_root)

    # the PROMISE12 dataset specifies the train and test splits
    # each case has 4 files associated with it
    # - CaseXX.mhd: the image metadata used for loading with ITKReader
    # - CaseXX.raw: image binary
    # - CaseXX_segmentation.mhd: label metadata
    # - CaseXX_segmentation.raw: label binary
    prefix = 'Case'
    mask_postfix = '_segmentation'
    data_paths = {
        'train': os.path.join(data_root, 'training_data'),
        'test': os.path.join(data_root, 'test_data')
    }
    # construct dataloaders for the training and testing data
    data_splits = {key: [] for key in SPLIT_NAMES}
    modality_info = {key: [] for key in SPLIT_NAMES}
    index = 0
    for split, path in data_paths.items():
        # find all paths that end with .mhd
        for image_path in sorted(glob.glob(os.path.join(path, f'{prefix}??.mhd'))):
            # find the corresponding label path
            label_path = image_path.replace('.mhd', f'{mask_postfix}.mhd')
            image_load = load_callback(promise12_image_loader, image_path)
            label_load = load_callback(promise12_label_loader, label_path)
            data_splits[split].append((str(index), image_load, label_load))
            modality_info[split].append('1') # MRI
            index += 1
    # split the training data into train and validation
    train_split, val_split = two_way_split(
        data_splits['train'], modality_info['train'], val_ratio=val_ratio)
    data_splits['train'] = train_split[0]
    modality_info['train'] = train_split[1]
    data_splits['val'] = val_split[0]
    modality_info['val'] = val_split[1]

    return data_splits, modality_info, promise_labels