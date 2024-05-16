"""
Dataset utilities for the AMOS dataset.
defines:
- amos_image_loader
- amos_label_loader
- amos_data_download
- parse_amos: main function to parse the AMOS dataset
"""

import os
import zipfile as zf
import json
import requests
import csv
from typing import Dict, List, Tuple, Callable
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Compose,
    ToNumpy
)

from util import load_callback, SPLIT_NAMES

def amos_data_download(data_root: str) -> None:
    """
        Download the AMOS dataset and the metadata,
        perpare the directory structure.
        OR do nothing if the data is already downloaded.
    """
    amos_data_url = 'https://zenodo.org/records/7262581/files/amos22.zip'
    amos_metadata_url = 'https://zenodo.org/records/7262581/files/labeled_data_meta_0000_0599.csv'

    # we expect data_root to point to AMOS_2022 directory (which may or may not exist)
    if os.path.basename(data_root) == 'amos22':
        data_root = os.path.dirname(data_root)
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    # download the data
    amos_data_path = os.path.join(data_root, 'amos22.zip')
    if not os.path.exists(amos_data_path):
        print('Downloading AMOS data...')
        response = requests.get(amos_data_url)
        with open(amos_data_path, 'wb') as f:
            f.write(response.content)
    # download the metadata
    metadata_path = os.path.join(data_root, 'labeled_data_meta_0000_0599.csv')
    if not os.path.exists(metadata_path):
        print('Downloading AMOS metadata...')
        response = requests.get(amos_metadata_url)
        with open(metadata_path, 'wb') as f:
            f.write(response.content)
    # unzip the data
    amos_data_dir = os.path.join(data_root, 'amos22')
    if not os.path.exists(amos_data_dir):
        print('Unzipping AMOS data...')
        with zf.ZipFile(amos_data_path, 'r') as z:
            z.extractall(data_root)
    # delete the zip file
    if os.path.exists(amos_data_path):
        os.remove(amos_data_path)
    return data_root

AMOS_MACHINE_TO_MODALITY = {
    'Ingenia': '1',
    'Optima CT660': '0',
    'Brilliance16': '0',
    'SIGNA HDe': '1',
    'Achieva': '1',
    'Optima CT540': '0',
    'SOMATOM Force': '0',
    'Aquilion ONE': '0',
    'Prisma': '1',
}

amos_image_loader = Compose(
    [
        LoadImage(),
        EnsureChannelFirst(channel_dim="no_channel"),
        ToNumpy()
    ]
)
amos_label_loader = amos_image_loader

def parse_amos(data_root: str) -> Tuple[Dict[str, List[Tuple[str, Callable]]],
                                        Dict[str, List[str]],
                                        List[str]]:
    """
        Parse the metadata for the AMOS dataset and return the data
        split as a dictionary.
    """
    # first make sure the data exists
    data_root = amos_data_download(data_root)

    prefix = 'amos_'
    # format of the files is
    # 'amos22/[images_[Tr|Va|Ts]|labels[Tr|Va|Ts]]/<prefix><amos_id>.nii.gz' in the data root
    # data root contains a dataset.json and its parent
    # (hopefully) contains labeled_data_meta_0000_0599.csv

    # load JSON
    dataset_json_path = os.path.join(data_root, 'amos22', 'dataset.json')
    if not os.path.isfile(dataset_json_path):
        raise ValueError('dataset.json not found in data root')
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)

    # Load the CSV file
    metadata_path = os.path.join(data_root, 'labeled_data_meta_0000_0599.csv')
    if not os.path.isfile(metadata_path):
        raise ValueError(
            'metadata file not found in {data_root}/labeled_data_meta_0000_0599.csv')
    with open(metadata_path, 'r') as f:
        metadata = csv.DictReader(f)
        metadata = {dp['amos_id']: dp for dp in metadata}

    data_paths = {
        'train': {'images': os.path.join(data_root, 'amos22', 'imagesTr'),
                  'labels': os.path.join(data_root, 'amos22', 'labelsTr')},
        'val': {'images': os.path.join(data_root, 'amos22', 'imagesVa'),
                'labels': os.path.join(data_root, 'amos22', 'labelsVa')},
        'test': {'images': os.path.join(data_root, 'amos22', 'imagesTs'),
                 'labels': os.path.join(data_root, 'amos22', 'labelsTs')}
    }

    #  create splits dictionary
    data_splits = {key: [] for key in SPLIT_NAMES}
    modality_info = {key: [] for key in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        images_dir = data_paths[split]['images']
        labels_dir = data_paths[split]['labels']
        images_list = sorted(os.listdir(images_dir))
        labels_list = sorted(os.listdir(labels_dir))
        for img, lab in zip(images_list, labels_list):
            img_id = img.split('.')[0][len(prefix):]
            modality = AMOS_MACHINE_TO_MODALITY[metadata[img_id]
                                                ['Manufacturer\'s Model Name']]
            data_splits[split].append((str(img_id),
                                       load_callback(
                                           amos_image_loader, os.path.join(images_dir, img)),
                                       load_callback(amos_image_loader, os.path.join(labels_dir, lab))))
            modality_info[split].append(modality)
    # check that the lengths are correct
    assert (len(data_splits['train']) == dataset_json['numTraining']), \
        f'number of training samples mismatch.' +\
        ' found: {len(data_splits["train"])} expected: {dataset_json["numTraining"]}'
    assert (len(data_splits['val']) == dataset_json['numValidation']), \
        f'number of validation samples mismatch.' +\
        ' found: {len(data_splits["val"])} expected: {dataset_json["numValidation"]}'
    assert (len(data_splits['test']) == dataset_json['numTest']), \
        f'number of test samples mismatch.' +\
        ' found: {len(data_splits["test"])} expected: {dataset_json["numTest"]}'

    # obtain list of labels and corr. classes
    classes = dataset_json['labels']

    return data_splits, modality_info, classes
