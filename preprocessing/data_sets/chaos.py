"""
Dataset utilities for the CHAOS dataset.
defines:
- chaos_image_loader
- chaos_label_loader
- chaos_data_download
"""

import os
from typing import List, Dict, Tuple, Callable
import numpy as np
import requests
import zipfile as zf
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Compose,
    ToNumpy
)

from util import load_callback, SPLIT_NAMES

def chaos_data_download(data_root: str) -> str:
    """
        Download the CHAOS dataset and the metadata,
        perpare the directory structure.
        OR do nothing if the data is already downloaded.
        Returns the path to the extracted data.
    """
    chaos_data_url = 'https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip'

    # data_root may point to Train_Sets directory (which may or may not exist)
    if os.path.basename(data_root) == 'Train_Sets':
        data_root = os.path.dirname(data_root)
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    chaos_data_path = os.path.join(data_root, 'CHAOS_Train_Sets.zip')
    chaos_data_dir = os.path.join(data_root, 'Train_Sets')

    if not os.path.exists(chaos_data_dir):
        # download the data
        if not os.path.exists(chaos_data_path):
            print('Downloading CHAOS data...')
            r = requests.get(chaos_data_url, allow_redirects=True)
            with open(chaos_data_path, 'wb') as f:
                f.write(r.content)
        # extract the data
        print('Extracting CHAOS data...')
        with zf.ZipFile(chaos_data_path, 'r') as z:
            z.extractall(data_root)
        # remove the zip file
        if os.path.exists(chaos_data_path):
            os.remove(chaos_data_path)

    return chaos_data_dir

chaos_classes = {
    '1': 'Liver',
    '2': 'Right kidney',
    '3': 'Left kidney',
    '4': 'Spleen'
}
# represent the intervals
chaos_cls_intervals = {
    '1': (55/255, 70/255),
    '2': (110/255, 135/255),
    '3': (175/255, 200/255),
    '4': (240/255, 255/255)
}

def chaos_image_loader(path):
    img_loader = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(channel_dim="no_channel"),
            ToNumpy()
        ]
    )
    img = img_loader(path)
    # transpose along the 2nd and 3rd axis
    img = np.transpose(img, (0, 2, 1, 3))
    return img


def chaos_label_loader(path, is_ct: bool):
    """
        CHAOS labels come in slices in PNG format. We need to stack them.
        In addition, the MRI masks are defined over invervals, so we need to
        discretize them as well.
        CT masks are binary, so we don't need to do anything.
    """
    label_array = []
    for slc in sorted(os.listdir(path)):
        label_img = plt.imread(os.path.join(path, slc))
        if is_ct:
            label_img[label_img > 0] = 1
        else:
            for c in chaos_classes.keys():
                # set the pixels in the given interval to the class
                label_img[(label_img >= chaos_cls_intervals[c][0]) &
                          (label_img <= chaos_cls_intervals[c][1])] = float(c) * -1
            label_img[label_img < 0] *= -1  # flip sign back
        label_array.append(label_img)
    label_array = np.stack(label_array, axis=-1).astype(np.int32)
    return ToNumpy()(EnsureChannelFirst(channel_dim="no_channel")(label_array))


def parse_chaos(data_root: str,
                test_ratio: float,
                val_ratio: float) -> Tuple[Dict[str, List[Tuple[int, Callable]]],
                                           Dict[str, List[str]],
                                           List[str]]:
    """
        Parse the metadata for the CHAOS dataset and return the data
        split as a dictionary.
    """
    # first make sure the data exists
    data_root = chaos_data_download(data_root)

    # CHAOS data has two directories at the root and no config files
    # CT directory contains CT images, in DICOM format and gt masks in PNG format
    # MR directory contains MRI images, in DICOM format and gt masks in PNG format
    # both the images and the masks are saved as <slice_number>.dcm/png, in an indexed directory

    # create splits dictionary
    data_splits = {key: [] for key in SPLIT_NAMES}
    modality_info = {key: [] for key in SPLIT_NAMES}
    splits_list = []
    mod_list = []
    # parse CT data
    ct_data_root = os.path.join(data_root, 'CT')
    for idx in os.listdir(ct_data_root):
        name = f'ct_{idx}'
        image_path = os.path.join(ct_data_root, idx, 'DICOM_anon')
        label_path = os.path.join(ct_data_root, idx, 'Ground')
        # construct loaders
        image_load = load_callback(chaos_image_loader, image_path)
        label_load = load_callback(chaos_label_loader,
                                   **{'path': label_path, 'is_ct': True})
        # append to the splits
        splits_list.append((name, image_load, label_load))
        mod_list.append('0')
    # create test split
    test_idx = np.random.choice(len(splits_list), int(
        test_ratio*len(splits_list)), replace=False)
    test_idx = set(test_idx)
    data_splits['test'] = [splits_list[idx] for idx in test_idx]
    modality_info['test'] = [mod_list[idx] for idx in test_idx]
    # remove the test splits
    rest_idx = set(range(len(splits_list))) - test_idx
    # create val split
    val_idx = np.random.choice(list(rest_idx), int(
        val_ratio*len(splits_list)), replace=False)
    val_idx = set(val_idx)
    data_splits['val'] = [splits_list[idx] for idx in val_idx]
    modality_info['val'] = [mod_list[idx] for idx in val_idx]
    # create train split
    train_idx = rest_idx - val_idx
    data_splits['train'] = [splits_list[idx] for idx in train_idx]
    modality_info['train'] = [mod_list[idx] for idx in train_idx]

    splits_list.clear()
    mod_list.clear()

    ###########
    # process MRI data
    mr_data_root = os.path.join(data_root, 'MR')
    for idx in os.listdir(mr_data_root):
        # T1DUAL subdirectory has InPhase and OutPhase subdirectories, both have the same mask
        # mask first
        label_path_dual = os.path.join(mr_data_root, idx, 'T1DUAL', 'Ground')
        label_loader_dual = load_callback(chaos_label_loader,
                                          **{'path': label_path_dual, 'is_ct': False})
        # InPhase
        image_path_in = os.path.join(
            mr_data_root, idx, 'T1DUAL', 'DICOM_anon', 'InPhase')
        image_loader_in = load_callback(chaos_image_loader, image_path_in)
        # OutPhase
        image_path_out = os.path.join(
            mr_data_root, idx, 'T1DUAL', 'DICOM_anon', 'OutPhase')
        image_loader_out = load_callback(chaos_image_loader, image_path_out)

        # T2SPIR subdirectory has a single mask
        # mask first
        label_path_spir = os.path.join(mr_data_root, idx, 'T2SPIR', 'Ground')
        label_loader_spir = load_callback(chaos_label_loader,
                                          **{'path': label_path_spir, 'is_ct': False})
        # image
        image_path_spir = os.path.join(
            mr_data_root, idx, 'T2SPIR', 'DICOM_anon')
        image_loader_spir = load_callback(chaos_image_loader, image_path_spir)
        # append, while keeping the in and out phase images together.
        # they come from the same patient so its not good to have them in different splits
        splits_list.append(((f'mr_{idx}_in', image_loader_in, label_loader_dual),
                            (f'mr_{idx}_out', image_loader_out,
                             label_loader_dual),
                            (f'mr_{idx}_spir', image_loader_spir, label_loader_spir)))
        mod_list.append(('1', '1', '1'))
    # create test split
    test_idx = np.random.choice(len(splits_list), int(
        test_ratio*len(splits_list)), replace=False)
    test_idx = set(test_idx)
    data_splits['test'].extend(
        [split for idx in test_idx for split in splits_list[idx]])
    modality_info['test'].extend(
        [mod for idx in test_idx for mod in mod_list[idx]])
    # remove the test splits
    rest_idx = set(range(len(splits_list))) - test_idx
    # create val split
    val_idx = np.random.choice(list(rest_idx), int(
        val_ratio*len(splits_list)), replace=False)
    val_idx = set(val_idx)
    data_splits['val'].extend(
        [split for idx in val_idx for split in splits_list[idx]])
    modality_info['val'].extend(
        [mod for idx in val_idx for mod in mod_list[idx]])
    # create train split
    train_idx = rest_idx - val_idx
    data_splits['train'].extend(
        [split for idx in train_idx for split in splits_list[idx]])
    modality_info['train'].extend(
        [mod for idx in train_idx for mod in mod_list[idx]])

    return data_splits, modality_info, chaos_classes
