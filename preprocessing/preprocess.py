import numpy as np
import scipy.sparse as sp
import os
import json
import multiprocessing
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
)

from cli import parse_arguments
from splits import (SPLIT_NAMES, MODALITY_MAPPING,
                    parse_amos, parse_chaos)
from process_modality import process_ct_image, process_mr_image

# image loader that we use for loading the images
# supports both nifti and dicom
img_loader = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label'], channel_dim="no_channel"),
        # Orientationd(keys=['image', 'label'], axcodes="RAS"),
    ]
)

def preprocess_image(info):
    """
        Load the image and label and save them to the save_path
    """
    name, img_loader, label_loader, modality, label_map, save_path = info
    # check if the case already exists
    case_path = os.path.join(save_path, name)
    os.makedirs(case_path, exist_ok=True)
    if os.path.exists(os.path.join(case_path, 'image.npy')):
        print(f'{name} already exists, skipping')
        return

    # load the image and label
    image_ndarray = img_loader() # prepared in splits.py
    label_ndarray = label_loader() # prepared in splits.py
    # normalize (and transform) the image
    if MODALITY_MAPPING[modality] == 'CT':
        image_ndarray = process_ct_image(image_ndarray)
    elif MODALITY_MAPPING[modality] == 'MRI':
        image_ndarray = process_mr_image(image_ndarray)
    else:
        raise ValueError(f'Unknown modality: {modality}')
    # split the labels into separate classes and stack them
    label_ndarray = np.array(label_ndarray).squeeze()
    gt_masks = []
    for requested_id, dataset_id in label_map.items():
        gt_mask = np.zeros_like(label_ndarray)
        if dataset_id == -1:
            print(f'{name} zero class ==> {requested_id}')
            gt_masks.append(gt_mask)
            continue
        gt_mask[label_ndarray == int(dataset_id)] = 1
        gt_masks.append(gt_mask)
    gt_masks = np.stack(gt_masks).astype(np.int32)

    print(name, 'ct gt <--> ', image_ndarray.shape, gt_masks.shape)
    # save the image and label
    np.save(os.path.join(case_path, 'image.npy'), image_ndarray)
    allmatrix_sp = sp.csr_matrix(gt_masks.reshape(gt_masks.shape[0], -1))
    sp.save_npz(os.path.join(case_path, 'label.npz'), allmatrix_sp)
    print(f'{name} saved')

def run():
    args = parse_arguments()

    # SEED
    np.random.seed(args.seed)

    # path to save stuff
    save_path = os.path.join(args.save_root, args.dataset_code)
    os.makedirs(save_path, exist_ok=True)
    # list of files that already exist
    existing_files = os.listdir(save_path)
    # split the data into training, validation and testing
    if args.dataset_type == 'AMOS':
        data_splits, modality_info, classes = parse_amos(args.dataset_root)
    elif args.dataset_type == 'CHAOS':
        data_splits, modality_info, classes = parse_chaos(args.dataset_root,
                                                          args.test_ratio,
                                                          args.val_ratio)
    else:
        raise ValueError(f'Unknown dataset code: {args.dataset_type}')
    # obtain (required_class_id, configured_class_id) mapping for the ground truth labels
    # required_class_id is the index of the list of categories in the arguments
    # configured_class_id is the index of the class in the dataset
    label_map = {idx+1: -1 for idx in range(len(args.classes))}
    label_map[0] = 0 # background class
    for idx, label in enumerate(args.classes):
        # find the corresponding class in the dataset
        dataset_class_id = -1
        for dataset_cls_id, dataset_class in classes.items():
            if dataset_class.lower() == label.lower():
                dataset_class_id = dataset_cls_id
                break
        label_map[idx+1] = dataset_class_id

    # prepare the data for multiprocessing
    # create a list of (image_name, image_path, label_path, modality) tuples
    linear_data = []
    for split in SPLIT_NAMES:
        for info, modality in zip(data_splits[split], modality_info[split]):
            linear_data.append((*info, modality, label_map, save_path))
    # create a pool of workers
    for data in linear_data:
        preprocess_image(data)
    # with multiprocessing.Pool(args.num_workers) as pool:
    #     # load the images and labels
    #     pool.map(preprocess_image, linear_data)
    # save JSON metadata
    output_meta = {
        'name': args.dataset_type or args.dataset_code,
        'description': args.dataset_code,
        'author': 'N/A',
        'reference': 'N/A',
        'licence': 'N/A',
        'release': 'N/A',
        'tensorImageSize': '4D',
        'modality': MODALITY_MAPPING,
        'labels': {str(idx): label for idx, label in enumerate(args.classes)},
        'numTraining': len(data_splits['train']),
        'numValidation': len(data_splits['val']),
        'numTest': len(data_splits['test']),
        'training': [{'image': f'{info[0]}/image.npy',
                      'label': f'{info[0]}/label.npz',
                      'modality': modality}
                     for info, modality in zip(data_splits['train'], modality_info['train'])],
        'validation': [{'image': f'{info[0]}/image.npy',
                        'label': f'{info[0]}/label.npz', 
                        'modality': modality}
                       for info, modality in zip(data_splits['val'], modality_info['val'])],
        'test': [{'image': f'{info[0]}/image.npy',
                  'label': f'{info[0]}/label.npz',
                  'modality': modality}
                 for info, modality in zip(data_splits['test'], modality_info['test'])],
    }
    # save the metadata
    with open(os.path.join(save_path, f'dataset.json'), 'w') as f:
        json.dump(output_meta, f, indent=2)
    print(f'dataset.json saved at ', save_path)
    print('Done!')

if __name__ == '__main__':
    run()