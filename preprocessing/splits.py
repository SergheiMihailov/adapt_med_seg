from typing import List, Dict, Tuple, Callable
import pydicom
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Compose,
    ToNumpy
)
from monai.transforms.spatial.functional import flip
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
class load_callback:
    def __init__(self, loader: Callable, *args, **kwargs) -> None:
        self.loader = loader
        self.args = args
        self.kwargs = kwargs
    def __call__(self):
        return self.loader(*self.args, **self.kwargs)

img_loader_amos = Compose(
    [
        LoadImage(),
        EnsureChannelFirst(channel_dim="no_channel"),
        ToNumpy()
    ]
)

## CHAOS dataset loaders
# CT data has only liver segmentation masks (binary), while
# MR data has multiple masks, these are
#    Liver: 63 (55 <<< 70)
#    Right kidney: 126 (110 <<< 135)
#    Left kidney: 189 (175 <<< 200)
#    Spleen: 252 (240 <<< 255)
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
            label_img[label_img < 0] *= -1 # flip sign back
        label_array.append(label_img)
    label_array = np.stack(label_array, axis=-1).astype(np.int32)
    return ToNumpy()(EnsureChannelFirst(channel_dim="no_channel")(label_array))


def parse_amos(data_root: str) -> Tuple[Dict[str, List[Tuple[str,Callable]]],
                                        Dict[str, List[str]],
                                        List[str]]:
    """
        Parse the metadata for the AMOS dataset and return the data
        split as a dictionary.
    """
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
        raise ValueError('metadata file not found in {data_root}/../labeled_data_meta_0000_0599.csv')
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

    # create splits dictionary
    data_splits = {key: [] for key in SPLIT_NAMES}
    modality_info = {key: [] for key in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        images_dir = data_paths[split]['images']
        labels_dir = data_paths[split]['labels']
        images_list = sorted(os.listdir(images_dir))
        labels_list = sorted(os.listdir(labels_dir))
        for img, lab in zip(images_list, labels_list):
            img_id = img.split('.')[0][len(prefix):]
            modality = AMOS_MACHINE_TO_MODALITY[metadata[img_id]['Manufacturer\'s Model Name']]
            data_splits[split].append((str(img_id),
                                       load_callback(img_loader_amos, os.path.join(images_dir, img)),
                                       load_callback(img_loader_amos, os.path.join(labels_dir, lab))))
            modality_info[split].append(modality)
    # check that the lengths are correct
    assert (len(data_splits['train']) == dataset_json['numTraining']),\
        f'number of training samples mismatch.'+\
            ' found: {len(data_splits["train"])} expected: {dataset_json["numTraining"]}'
    assert (len(data_splits['val']) == dataset_json['numValidation']),\
        f'number of validation samples mismatch.' +\
        ' found: {len(data_splits["val"])} expected: {dataset_json["numValidation"]}'
    assert (len(data_splits['test']) == dataset_json['numTest']),\
        f'number of test samples mismatch.' +\
        ' found: {len(data_splits["test"])} expected: {dataset_json["numTest"]}'

    # obtain list of labels and corr. classes
    classes = dataset_json['labels']

    return data_splits, modality_info, classes

def parse_chaos(data_root: str,
                test_ratio: float,
                val_ratio: float) -> Tuple[Dict[str, List[Tuple[int,Callable]]],
                                         Dict[str, List[str]],
                                         List[str]]:
    """
        Parse the metadata for the CHAOS dataset and return the data
        split as a dictionary.
    """

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
    test_idx = np.random.choice(len(splits_list), int(test_ratio*len(splits_list)), replace=False)
    test_idx = set(test_idx)
    data_splits['test'] = [splits_list[idx] for idx in test_idx]
    modality_info['test'] = [mod_list[idx] for idx in test_idx]
    # remove the test splits
    rest_idx = set(range(len(splits_list))) - test_idx
    # create val split
    val_idx = np.random.choice(list(rest_idx), int(val_ratio*len(splits_list)), replace=False)
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
        image_path_spir = os.path.join(mr_data_root, idx, 'T2SPIR', 'DICOM_anon')
        image_loader_spir = load_callback(chaos_image_loader, image_path_spir)
        # append, while keeping the in and out phase images together.
        # they come from the same patient so its not good to have them in different splits
        splits_list.append(((f'mr_{idx}_in', image_loader_in, label_loader_dual),
                            (f'mr_{idx}_out', image_loader_out, label_loader_dual),
                            (f'mr_{idx}_spir', image_loader_spir, label_loader_spir)))
        mod_list.append(('1', '1', '1'))
    # create test split
    test_idx = np.random.choice(len(splits_list), int(test_ratio*len(splits_list)), replace=False)
    test_idx = set(test_idx)
    data_splits['test'].extend(
        [split for idx in test_idx for split in splits_list[idx]])
    modality_info['test'].extend([mod for idx in test_idx for mod in mod_list[idx]])
    # remove the test splits
    rest_idx = set(range(len(splits_list))) - test_idx
    # create val split
    val_idx = np.random.choice(list(rest_idx), int(val_ratio*len(splits_list)), replace=False)
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