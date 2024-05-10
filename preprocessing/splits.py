from typing import List, Dict, Tuple
import pydicom
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

def parse_amos(data_root: str) -> Tuple[Dict[str, List[Tuple[str,str,str]]],
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
                                       os.path.join(images_dir, img),
                                       os.path.join(labels_dir, lab)))
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

def parse_chaos(data_root: str) -> Tuple[Dict[str, List[Tuple[int,str,str,str]]],
                                         Dict[str, List[str]],
                                         List[str]]:
    """
        Parse the metadata for the CHAOS dataset and return the data
        split as a dictionary.
    """
    def stack_dicom_slices(image_path: str) -> np.ndarray:
        image_array = [
            pydicom.dcmread(
                os.path.join(image_path, os.listdir(img))).pixel_array
            for img in os.listdir(image_path)
        ]
        image_array = np.stack(image_array, axis=0)
        image_path = os.path.join(image_path, 'image.npy')
        return image_array, image_path
    def stack_mr_masks(label_path: str) -> np.ndarray:
        label_array = []
        for label in os.listdir(label_path):
            label_img = np.imread(os.path.join(label_path, label))
            for c in classes.keys():
                # set the pixels in the given interval to the class
                label_img[(label_img >= cls_intervals[c][0]) &
                          (label_img <= cls_intervals[c][1])] = -1 * c  # avoid conflict
            label_img *= -1  # flip sign back
            label_array.append(label_img)
        label_array = np.stack(label_array, axis=0)
        label_path = os.path.join(label_path, 'label.npz')
        return label_array, label_path

    # CHAOS data has two directories at the root and no config files
    # CT directory contains CT images, in DICOM format and gt masks in PNG format
    # MR directory contains MRI images, in DICOM format and gt masks in PNG format
    # both the images and the masks are saved as <slice_number>.dcm/png, in an indexed directory

    # CT data has only liver segmentation masks (binary), while
    # MR data has multiple masks, these are 
    #    Liver: 63 (55 <<< 70)
    #    Right kidney: 126 (110 <<< 135)
    #    Left kidney: 189 (175 <<< 200)
    #    Spleen: 252 (240 <<< 255)
    classes = {
        '1': 'Liver',
        '2': 'Right kidney',
        '3': 'Left kidney',
        '4': 'Spleen'
    }
    # represent the intervals
    cls_intervals = {
        '1': (55, 70),
        '2': (110, 135),
        '3': (175, 200),
        '4': (240, 255)
    }
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
        # stack the individual slices
        image_array, image_path = stack_dicom_slices(image_path)
        # convert label to numpy array. We do this to avoid having to handle intervals later
        label_array = []
        for label in os.listdir(label_path):
            label_img = np.imread(os.path.join(label_path, label))
            label_img[label_img > 0] = 1 # only liver
            label_array.append(label_img)
        label_array = np.stack(label_array, axis=0)
        label_path = os.path.join(label_path, 'label.npz')
        # save the pre-preprocessed image and label
        np.save(image_path, image_array)
        np.save(label_path, label_array)
        print(f'{name} image and label stacked and saved during read <--> {image_array.shape} {label_array.shape}')
        # append to the splits
        splits_list.append((name, image_path, label_path))
        mod_list.append('0')
    # create test split
    test_idx = np.random.choice(len(splits_list), int(0.2*len(splits_list)), replace=False)
    test_idx = set(test_idx)
    data_splits['test'] = [splits_list[idx] for idx in test_idx]
    modality_info['test'] = [mod_list[idx] for idx in test_idx]
    # remove the test splits
    rest_idx = set(range(len(splits_list))) - test_idx
    # create val split
    val_idx = np.random.choice(list(rest_idx), int(0.2*len(splits_list)), replace=False)
    val_idx = set(val_idx)
    data_splits['val'] = [splits_list[idx] for idx in val_idx]
    modality_info['val'] = [mod_list[idx] for idx in val_idx]
    # create train split
    train_idx = rest_idx - val_idx
    data_splits['train'] = [splits_list[idx] for idx in train_idx]
    modality_info['train'] = [mod_list[idx] for idx in train_idx]
    # sanity check
    assert len(data_splits['train']) + len(data_splits['val']) + len(data_splits['test']) == len(splits_list),\
        'splits do not add up'

    splits_list.clear()
    mod_list.clear()

    ###########
    # process MRI data
    mr_data_root = os.path.join(data_root, 'MR')
    for idx in os.listdir(mr_data_root):
        # T1DUAL subdirectory has InPhase and OutPhase subdirectories, both have the same mask
        # mask first
        label_path_dual = os.path.join(mr_data_root, idx, 'T1DUAL', 'Ground')
        label_array_dual, label_path_dual = stack_mr_masks(label_path_dual)
        # InPhase
        image_path_in = os.path.join(mr_data_root, idx, 'T1DUAL', 'InPhase')
        image_array_in, image_path_in = stack_dicom_slices(image_path)
        # OutPhase
        image_path_out = os.path.join(mr_data_root, idx, 'T1DUAL', 'OutPhase')
        image_array_out, image_path_out = stack_dicom_slices(image_path)

        # T2SPIR subdirectory has a single mask
        # mask first
        label_path_spir = os.path.join(mr_data_root, idx, 'T2SPIR', 'Ground')
        label_array_spir, label_path_spir = stack_mr_masks(label_path_spir)
        # image
        image_path_spir = os.path.join(mr_data_root, idx, 'T2SPIR', 'DICOM_anon')
        image_array_spir, image_path_spir = stack_dicom_slices(image_path_spir)
        # append, while keeping the in and out phase images together.
        # they come from the same patient so its not good to have them in different splits
        splits_list.append(((f'mr_{idx}_in', image_path_in, label_path_dual),
                            (f'mr_{idx}_out', image_path_out, label_path_dual),
                            (f'mr_{idx}_spir', image_path_spir, label_path_spir)))
        mod_list.append(('1', '1', '1'))
    # create test split
    test_idx = np.random.choice(len(splits_list), int(0.2*len(splits_list)), replace=False)
    test_idx = set(test_idx)
    data_splits['test'].extend(
        [split for idx in test_idx for split in splits_list[idx]])
    modality_info['test'].extend([mod for idx in test_idx for mod in mod_list[idx]])
    # remove the test splits
    rest_idx = set(range(len(splits_list))) - test_idx
    # create val split
    val_idx = np.random.choice(list(rest_idx), int(0.2*len(splits_list)), replace=False)
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
    # sanity check
    assert len(data_splits['train']) + len(data_splits['val']) + len(data_splits['test']) == len(splits_list),\
        'splits do not add up'

    return data_splits, modality_info, classes