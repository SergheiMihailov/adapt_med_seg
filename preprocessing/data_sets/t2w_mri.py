"""
Dataset for the 'T2 Weighted MRI images' dataset, from
https://zenodo.org/records/7676958

defines:
- t2w_mri_image_loader
- t2w_mri_label_loader
- t2w_mri_data_download
- parse_t2w_mri
"""
import tifffile as tiff
import zipfile as zf
import requests
import os
from glob import glob
import numpy as np
from typing import List, Dict, Tuple, Callable

from util import load_callback, SPLIT_NAMES, three_way_split

T2W_LABELS = {
    '1': 'prostate'
}

def t2w_mri_data_download(data_path: str) -> str:
    """
    Download the T2W MRI dataset and the metadata,
    prepare the directory structure.
    OR do nothing if the data is already downloaded.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    t2w_data_path = os.path.join(data_path, 'prostate_dataset.zip')
    t2w_data_dir = os.path.join(data_path, 'prostate_dataset')

    if not os.path.exists(t2w_data_dir):
        # download the data
        if not os.path.exists(t2w_data_path):
            print('Downloading T2W MRI data...')
            t2w_data_url = 'https://zenodo.org/records/7676958/files/prostate_dataset.zip'
            r = requests.get(t2w_data_url, allow_redirects=True)
            with open(t2w_data_path, 'wb') as f:
                f.write(r.content)
        # extract the data
        print('Extracting T2W MRI data...')
        with zf.ZipFile(t2w_data_path, 'r') as z:
            z.extractall(data_path)
        # remove the zip file
        if os.path.exists(t2w_data_path):
            os.remove(t2w_data_path)

    return t2w_data_dir

def t2w_mri_image_loader(image_path: str) -> np.ndarray:
    img = tiff.imread(image_path)
    img = np.transpose(img, (1, 2, 0))
    # add a channel dimension
    img = np.expand_dims(img, axis=0)
    return img

def t2w_mri_label_loader(label_path: str) -> np.ndarray:
    img = t2w_mri_image_loader(label_path)
    # there are two classes, periphery and transition zone
    # we will combine them into one label
    img[img > 0] = 1
    return img

def parse_t2w_mri(data_root: str,
                  test_ratio: float,
                  val_ratio: float) -> Tuple[Dict[str, List[Tuple[str, Callable]]],
                                             Dict[str, List[str]],
                                             List[str]]:
    """
        Parse the T2W MRI dataset.
        Returns a dictionary with keys 'train', 'val', 'test' and 'all',
        where each key maps to a list of tuples (image_path, loader_fn).
    """
    # download the data
    t2w_data_dir = t2w_mri_data_download(data_root)

    image_dir = os.path.join(t2w_data_dir, 'T2_IMAGES')
    label_dir = os.path.join(t2w_data_dir, 'LABELS')
    # the directory also has a CSV containing metadata but we ignore it
    prefix = 'P_'
    data_list = []
    modality_list = []
    for image_path in sorted(glob(image_dir)):
        # format is P_???.tiff
        # label format is P_???_SEG.tiff
        label_path = os.path.join(label_dir, os.path.basename(image_path).replace('.tiff', '_SEG.tiff'))
        img_id = int(os.path.basename(image_path).replace('.tiff', '')[len(prefix):])
        if not os.path.exists(label_path):
            print(f'Could not find corresponding label for {image_path}')
            continue
        image_loader = load_callback(image_path, t2w_mri_image_loader)
        label_loader = load_callback(label_path, t2w_mri_label_loader)
        data_list.append((img_id, image_loader, label_loader))
        modality_list.append('1')
    # create the splits
    data_splits = {key: [] for key in SPLIT_NAMES}
    modality_info = {key: [] for key in SPLIT_NAMES}
    train_split, val_split, test_split = three_way_split(
        data_list, modality_list, test_ratio=test_ratio, val_ratio=val_ratio)
    data_splits['train'] = train_split[0]
    modality_info['train'] = train_split[1]
    data_splits['val'] = val_split[0]
    modality_info['val'] = val_split[1]
    data_splits['test'] = test_split[0]
    modality_info['test'] = test_split[1]

    return data_splits, modality_info, T2W_LABELS
