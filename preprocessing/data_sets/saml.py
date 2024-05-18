"""
Dataset utilities for the SAML dataset.
defines:
- saml_image_loader
- saml_label_loader
- saml_data_download
- parse_saml: main function to parse the SAML dataset
"""
import gdown
import os
import numpy as np
from glob import glob
import zipfile
from typing import List, Dict, Tuple, Callable
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ToNumpy,
    Compose
)

from util import load_callback, SPLIT_NAMES, three_way_split

SAML_LABELS = {
    '1': 'prostate'
}

def saml_data_download(data_path: str) -> str:
    """
    Download the SAML dataset and the metadata,
    prepare the directory structure.
    OR do nothing if the data is already downloaded.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    saml_data_path = os.path.join(data_path, 'Processed_data_nii.zip')
    saml_data_dir = os.path.join(data_path, 'Processed_data_nii')

    if not os.path.exists(saml_data_dir):
        # download the data
        if not os.path.exists(saml_data_path):
            print('Downloading SAML data...')
            gdrive_url = 'https://drive.google.com/uc?id=1TtrjnlnJ1yqr5m4LUGMelKTQXtvZaru-'
            gdown.download(gdrive_url, saml_data_path, quiet=False)
        # extract the data
        print('Extracting SAML data...')
        with zipfile.ZipFile(saml_data_path, 'r') as z:
            z.extractall(data_path)
        # remove the zip file
        if os.path.exists(saml_data_path):
            os.remove(saml_data_path)

    return saml_data_dir

saml_image_loader = Compose([
    LoadImage(),
    EnsureChannelFirst(channel_dim="no_channel"),
    ToNumpy()
])
saml_label_loader = saml_image_loader


def parse_saml(data_path: str,
               test_ratio,
               val_ratio) -> Tuple[Dict[str, List[Tuple[str, Callable]]],
                                   Dict[str, List[str]],
                                   List[str]]:
    """
    Parse the SAML dataset.
    """
    # download the data
    data_path = saml_data_download(data_path)

    prefix = 'Case'
    # the root contains several subdirectories, each corresponding to a site,
    # each site contains several cases
    # we disregard the site information and treat all cases as one dataset
    image_paths = sorted(glob(os.path.join(data_path, '*', f'{prefix}??.nii.gz')))

    data_list = []
    modality_list = []
    for idx, image_path in enumerate(image_paths):
        # the label has the same name, but _Segmentation suffix
        label_path = image_path.replace('.nii.gz', '_Segmentation.nii.gz')
        if not os.path.exists(label_path):
            print(f'Label file {label_path} not found.')
            continue
        image_loader = load_callback(image_path, saml_image_loader)
        label_loader = load_callback(label_path, saml_label_loader)
        data_list.append((idx, image_loader, label_loader))
        modality_list.append('1')

    # split the data
    data_splits = {key: [] for key in SPLIT_NAMES}
    modality_info = {key: [] for key in SPLIT_NAMES}
    train_splits, val_splits, test_splits = three_way_split(
        data_list, modality_list, test_ratio=test_ratio, val_ratio=val_ratio)
    data_splits['train'] = train_splits[0]
    modality_info['train'] = train_splits[1]
    data_splits['val'] = val_splits[0]
    modality_info['val'] = val_splits[1]
    data_splits['test'] = test_splits[0]
    modality_info['test'] = test_splits[1]

    return data_splits, modality_info, SAML_LABELS
