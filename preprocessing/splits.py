from typing import List, Dict, Tuple
import numpy as np
import json
import csv
import os

SPLIT_NAMES = ['train', 'val', 'test']
MODALITY_MAPPING = {
    '0': 'CT',
    '1': 'MRI'
}
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

def split_data(data_root: str,
               data_path_list_all: List[Tuple[int,str,str,str]],
               modality: str,
               test_ratio: float,
               val_ratio: float) -> Tuple[Dict[str, List[Tuple[int,str,str,str]]],
                                          Dict[str, List[str]],
                                          None]:
    """
        Randomly split the data into training and testing sets,
        with the test_ratio determining the proportion of the data
        to be used for testing.
        
        Return a dictionary containing the training and testing data and
        a dictionary containing information about the modality.
    """
    if test_ratio < 0 or test_ratio > 1:
        raise ValueError('test_ratio must be between 0 and 1')
    if val_ratio < 0 or val_ratio > 1:
        raise ValueError('val_ratio must be between 0 and 1')
    np.random.shuffle(data_path_list_all)
    num_test = int(test_ratio * len(data_path_list_all))
    test_data = data_path_list_all[:num_test]
    num_val = int((len(data_path_list_all)-num_test) * val_ratio)
    val_data = data_path_list_all[num_test:num_test+num_val]
    train_data = data_path_list_all[num_test+num_val:]
    data_splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    modality_info = {
        'train' : [modality] * len(train_data),
        'val' : [modality] * len(val_data),
        'test' : [modality] * len(test_data)
    }
    return data_splits, modality_info, None # we don't know the classes

def parse_amos_metadata(data_root: str,
                        data_path_list_all: List[Tuple[int,str,str,str]]
                        ) -> Tuple[Dict[str, List[Tuple[int,str,str,str]]],
                                   Dict[str, List[str]],
                                   List[str]]:
    """
        Parse the metadata for the AMOS dataset and return the data
        split as a dictionary.
    """
    # format of the files is
    # '[images_[Tr|Va|Ts]|labels[Tr|Va|Ts]]/<prefix><amos_id>.nii.gz' in the data root
    prefix = 'amos_'
    # data root contains a dataset.json and its parent
    # (hopefully) contains labeled_data_meta_0000_0599.csv

    # load JSON
    dataset_json_path = os.path.join(data_root, 'dataset.json')
    if not os.path.isfile(dataset_json_path):
        raise ValueError('dataset.json not found in data root')
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)
    # create sets from the image and label paths in the dataset.json
    training_ids = set([os.path.basename(item['image']).split('.')[0][len(prefix):]
                        for item in dataset_json['train']])
    validation_ids = set([os.path.basename(item['image']).split('.')[0][len(prefix):]
                          for item in dataset_json['val']])
    test_ids = set([os.path.basename(item['image']).split('.')[0][len(prefix):]
                    for item in dataset_json['test']])

    # Load the CSV file
    parent_dir = os.path.dirname(data_root)
    metadata_path = os.path.join(parent_dir, 'labeled_data_meta_0000_0599.csv')
    if not os.path.isfile(metadata_path):
        raise ValueError('metadata file not found in {data_root}/../labeled_data_meta_0000_0599.csv')
    with open(metadata_path, 'r') as f:
        metadata = csv.DictReader(f)
        metadata = {dp['amos_id']: dp for dp in metadata}

    # create splits dictionary
    data_splits = {key: [] for key in SPLIT_NAMES}
    modality_info = {key: [] for key in SPLIT_NAMES}
    for idx, name, image_path, label_path in data_path_list_all:
        # we ignore idx, because it may not reflect the actual index
        amos_id = name[len(prefix):]
        modality = AMOS_MACHINE_TO_MODALITY[metadata[amos_id]['Manufacturer\'s Model Name']]
        if amos_id in training_ids:
            data_splits['train'].append((idx, name, image_path, label_path))
            modality_info['train'].append(modality)
        elif amos_id in validation_ids:
            data_splits['val'].append((idx, name, image_path, label_path))
            modality_info['val'].append(modality)
        elif amos_id in test_ids:
            data_splits['test'].append((idx, name, image_path, label_path))
            modality_info['test'].append(modality)
    # check that the lengths are correct
    assert (len(data_splits['train']) == len(training_ids)
            == len(modality_info['train']) == dataset_json['numTraining'])
    assert (len(data_splits['val']) == len(validation_ids)
            == len(modality_info['val']) == dataset_json['numValidation'])
    assert (len(data_splits['test']) == len(test_ids)
            == len(modality_info['test']) == dataset_json['numTest'])

    # obtain list of labels and corr. classes
    classes = dataset_json['labels']

    return data_splits, modality_info, classes

def parse_chaos_metadata(data_root: str,
                         data_path_list_all: List[Tuple[int,str,str,str]]
                         ) -> Tuple[Dict[str, List[Tuple[int,str,str,str]]],
                                    Dict[str, List[str]],
                                    List[str]]:
    """
        Parse the metadata for the CHAOS dataset and return the data
        split as a dictionary.
    """
    data_splits = {key: [] for key in SPLIT_NAMES}
    # TODO: implement this

def get_data_splits(data_root: str,
                    data_path_list_all: List[Tuple[int,str,str,str]],
                    dataset_type: str | None = None,
                    test_ratio: float | None = None) -> Tuple[Dict[str, List[Tuple[int,str,str,str]]],
                                                              Dict[str, List[str]],
                                                              List[str] | None]:
    if not os.path.isdir(data_root):
        raise ValueError('Invalid data root')
    if dataset_type is not None:
        if dataset_type == 'AMOS':
            return parse_amos_metadata(data_root, data_path_list_all)
        elif dataset_type == 'CHAOS':
            return parse_chaos_metadata(data_root, data_path_list_all)
        else:
            raise ValueError('Unknown dataset type')
    if test_ratio is not None:
        return split_data(data_root, data_path_list_all, test_ratio)