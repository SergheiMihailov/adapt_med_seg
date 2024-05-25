"""
Dataset utilities for the Brats2021 dataset.
Defines:
- brats2021_image_loader
- brats2021_label_loader
- parse_brats2021: main function to parse the Brats2021 dataset
"""

import os
import tarfile as tf
import kaggle
import requests
from typing import Dict, List, Tuple
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Compose,
    ToNumpy
)

from util import SPLIT_NAMES, three_way_split, load_callback

def brats2021_data_download(data_root: str) -> str:
    """
    Download the Brats2021 dataset and the metadata,
    prepare the directory structure.
    OR do nothing if the data is already downloaded.

    Returns the path to the extracted data.
    """
    # kaggle datasets download -d dschettler8845/brats-2021-task1
    brats2021_data_path = os.path.join(data_root, 'BraTS2021_Training_Data.tar')
    brats2021_data_dir = os.path.join(data_root, 'brats2021')
    # data_root is where we extract the data
    os.makedirs(data_root, exist_ok=True)

    # download and extract the data
    if not os.path.exists(brats2021_data_dir):
        kaggle.api.dataset_download_cli('dschettler8845/brats-2021-task1',
                                        path=data_root, force=False, unzip=True)
        # extract the data
        print('Extracting Brats2021 data...')
        with tf.open(brats2021_data_path) as tar:
            tar.extractall(brats2021_data_dir)
        # remove the zip file
        if os.path.exists(brats2021_data_path):
            os.remove(brats2021_data_path)
    else:
        print('Brats2021 data already exists.')

    return brats2021_data_dir


BRATS2021_LABELS = {
    # NCR: Non-Contrast-Enhancing Tumor Core, ET: Enhancing Tumor, ED: Edema
    '1': 'NCR',
    '2': 'ET',
    '3': 'ED'
}

brats2021_image_loader = Compose(
    [
        LoadImage(),
        EnsureChannelFirst(channel_dim="no_channel"),
        ToNumpy()
    ])

def brats2021_label_loader(label_path: str):
    gt_mask = brats2021_image_loader(label_path)
    # replace label 4 (ED) with label 3 to have contiguous labels
    gt_mask[gt_mask == 4] = 3
    return gt_mask

def parse_brats2021(data_root: str, val_ratio: float = 0.2, test_ratio: float = 0.4) -> Tuple[Dict, Dict, List]:
    """
    Parse the Brats2021 dataset.
    validation ratio is not consider since dataset 
    its already divided in train and val 
    """
    data_root = brats2021_data_download(data_root)

    # each sample is in their designated directories formatted as
    # BraTS2021_?????/BraTS2021_?????_[type].nii.gz
    # where type is one of flair, t1, t1ce, t2, seg
    # seg is the ground truth segmentation
    prefix = 'BraTS2021_'

    data_splits = {key: [] for key in SPLIT_NAMES}
    modality_info = {key: [] for key in SPLIT_NAMES}
    classes = ['flair', 't1', 't1ce', 't2']

    def process_sample(sample_dir: str, sample_id: str):
        sample = {}
        for type_ in classes:
            modality_path = os.path.join(sample_dir, f'{sample_id}_{type_}.nii.gz')
            sample[type_] = modality_path
        label_path = os.path.join(sample_dir, f'{sample_id}_seg.nii.gz')
        sample['label'] = label_path
        return sample

    data_list = []
    modality_list = []
    # iterate over the data directory
    for sample_dir in os.listdir(data_root):
        if not sample_dir.startswith(prefix):
            continue
        sample_id = sample_dir[len(prefix):]
        sample = process_sample(os.path.join(data_root, sample_dir), sample_id)
        label_path = os.path.join(data_root, sample_dir, sample.pop('label'))
        label_loader = load_callback(brats2021_label_loader, label_path)
        for type_ in classes:
            image_path = os.path.join(data_root, sample_dir, sample[type_])
            image_loader = load_callback(brats2021_image_loader, image_path)
            data_list.append((sample_id, image_loader, label_loader))
            modality_list.append('1') # MRI

    train_set, val_set, test_set = three_way_split(data_list, modality_list,
                                                   val_ratio=val_ratio,
                                                   test_ratio=test_ratio)
    data_splits['train'] = train_set[0]
    modality_info['train'] = train_set[1]
    data_splits['val'] = val_set[0]
    modality_info['val'] = val_set[1]
    data_splits['test'] = test_set[0]
    modality_info['test'] = test_set[1]

    print('train:', len(data_splits['train']),
          'val:', len(data_splits['val']),
          'test:', len(data_splits['test']))

    return data_splits, modality_info, BRATS2021_LABELS
