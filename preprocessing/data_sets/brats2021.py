"""
Dataset utilities for the Brats2021 dataset.
Defines:
- brats2021_image_loader
- brats2021_label_loader
- parse_brats2021: main function to parse the Brats2021 dataset
"""

import os
import zipfile as zf
import requests
from typing import Dict, List, Tuple
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Compose,
    ToNumpy
)

from util import SPLIT_NAMES, two_way_split

BRATS2021_LABELS = {
    '1': 'brain'
}

def brats2021_data_download(data_root: str) -> str:
    """
    Download the Brats2021 dataset and the metadata,
    prepare the directory structure.
    OR do nothing if the data is already downloaded.

    Returns the path to the extracted data.
    """
    brats2021_data_url = 'https://www.kaggle.com/api/v1/datasets/download/syedsajid/brats2021?datasetVersionNumber=3'
    brats2021_data_path = os.path.join(data_root, 'brats2021.zip')
    brats2021_data_dir = os.path.join(data_root, 'brats2021')
    # data_root is where we extract the data
    os.makedirs(data_root, exist_ok=True)

    # download and extract the data
    if not os.path.exists(brats2021_data_dir):
        if not os.path.exists(brats2021_data_path):
            print('Downloading Brats2021 data...')
            r = requests.get(brats2021_data_url, allow_redirects=True)
            with open(brats2021_data_path, 'wb') as f:
                f.write(r.content)
        # extract the data
        print('Extracting Brats2021 data...')
        with zf.ZipFile(brats2021_data_path, 'r') as z:
            z.extractall(data_root)
        # remove the zip file
        if os.path.exists(brats2021_data_path):
            os.remove(brats2021_data_path)
    else:
        print('Brats2021 data already exists.')

    return data_root


def brats2021_image_loader(file_path: str):
    return Compose([LoadImage(image_only=True), EnsureChannelFirst(), ToNumpy()])(file_path)

def brats2021_label_loader(file_path: str):
    return Compose([LoadImage(image_only=True), EnsureChannelFirst(), ToNumpy()])(file_path)

def parse_brats2021(data_root: str, val_ratio: float = 0.2, test_ratio: float = 0.4) -> Tuple[Dict, Dict, List]:
    """
    Parse the Brats2021 dataset.
    validation ratio is not consider since dataset 
    its already divided in train and val 
    """
    brats2021_data_download(data_root)

    data_paths = {
        'train': os.path.join(data_root, 'MICCAI_FeTS2021_TrainingData', 'MICCAI_FeTS2021_TrainingData'),
        'val': os.path.join(data_root, 'MICCAI_FeTS2021_ValidationData', 'MICCAI_FeTS2021_ValidationData')
    }

    data_splits = {key: [] for key in SPLIT_NAMES}
    modality_info = {key: [] for key in SPLIT_NAMES}
    classes = ['flair', 't1', 't1ce', 't2']

    def process_split(split_name, data_dir, has_labels=True):
        for subject in sorted(os.listdir(data_dir)):
            subject_path = os.path.join(data_dir, subject)
            if not os.path.isdir(subject_path):
                continue
            images = {modality: os.path.join(subject_path, f'{subject}_{modality}.nii') for modality in classes}
            labels = os.path.join(subject_path, f'{subject}_seg.nii') if has_labels else None
            data_splits[split_name].append((subject, images, labels))
            modality_info[split_name].append(classes)

    process_split('train', data_paths['train'])
    process_split('val', data_paths['val'], has_labels=False)

    # Split the validation data into val and test
    val_data = data_splits['val']
    val_modality_info = modality_info['val']
    
    val_split, test_split = two_way_split(val_data, val_modality_info, val_ratio=test_ratio)
    data_splits['val'] = val_split[0]
    modality_info['val'] = val_split[1]
    data_splits['test'] = test_split[0]
    modality_info['test'] = test_split[1]

    print('train:', len(data_splits['train']),
          'val:', len(data_splits['val']),
          'test:', len(data_splits['test']))

    return data_splits, modality_info, BRATS2021_LABELS
