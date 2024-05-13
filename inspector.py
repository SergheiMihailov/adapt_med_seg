import numpy as np
from glob import glob
import argparse
import sys
import os
import ast
from scipy import sparse
import matplotlib.pyplot as plt

from SegVol.model_segvol_single import SegVolProcessor
# compare CHAOS pre-processed data-files

def load_mask(dir_path):
    gt_npy_path = glob(os.path.join(dir_path, 'mask_*.npz'))[0].split('/')[-1]
    gt_shape = ast.literal_eval(gt_npy_path.split("_")[-1].replace(".npz", ""))
    gt_npy = sparse.load_npz(os.path.join(dir_path, gt_npy_path)).toarray().reshape(gt_shape)
    return gt_npy


def load_image_mask(mypath, theirpath: str = None):
    my_image = np.load(os.path.join(
        mypath, 'image.npy')).astype(np.float32)
    my_mask = load_mask(mypath).astype(np.float32)
    if theirpath is not None:
        their_image = np.load(os.path.join(
            theirpath, 'image.npy')).astype(np.float32)
        their_mask = load_mask(theirpath).astype(np.float32)
        return my_image, my_mask, their_image, their_mask
    else:
        return my_image, my_mask, None, None

def load_all_image_mask(data_idx, mypath, prefix, theirpath):
    my_paths = sorted(glob(os.path.join(mypath, f'{prefix}')))
    if theirpath is not None:
        their_paths = [os.path.join(theirpath, p) for p in os.listdir(theirpath)\
            if not p.endswith('.json')]
    else:
        their_paths = [None] * len(my_paths)
    their_paths = sorted(their_paths) if theirpath is not None else their_paths
    if len(my_paths) == 0:
        raise ValueError(f'No matches found in {mypath} with prefix {prefix}')
    if data_idx is not None:
        my_paths = [my_paths[data_idx]]
        their_paths = [their_paths[data_idx]]
    image_mask_list = []
    print('my_paths', my_paths)
    print('their_paths', their_paths)
    for my_path, their_path in zip(my_paths, their_paths):
        my_image, my_mask, their_image, their_mask = load_image_mask(my_path, their_path)
        my_name = my_path.split('/')[-1]
        their_name = their_path.split('/')[-1] if their_path is not None else None
        if their_path is not None:
            print('my_image', my_image.shape, 'my_mask', my_mask.shape)
            print('their_image', their_image.shape, 'their_mask', their_mask.shape)

            print('average difference in image:', np.mean(np.abs(my_image - their_image)))
            print('std of image difference:', np.std(my_image - their_image))
            print('number of mask disagreements:', np.sum(my_mask[0] != their_mask[0]))
            # assert np.allclose(my_image, their_image)
            # assert np.allclose(my_mask[0], their_mask[0])
            print('-------------------------')
        image_mask_list.append((my_name, my_image, my_mask, their_name, their_image, their_mask))
    return image_mask_list

def show_image_mask(data_index, data_path, prefix, cls_idx=1):
    image_mask_list = load_all_image_mask(data_index, data_path, prefix, None)
    print('loaded', len(image_mask_list), 'images and masks')
    my_name, my_image, my_mask, _, _, _ = image_mask_list[0]
    print('my_image', my_image.shape, 'my_mask', my_mask.shape)
    fig, ax = plt.subplots(1, 1)
    slice_index = 0
    data_idx = 0

    ax.set_title(f'{my_name} s:{slice_index}')
    im0 = ax.imshow(my_image[0, :, :, slice_index].squeeze(), cmap='gray')
    im1 = ax.imshow(
        my_mask[cls_idx, :, :, slice_index].squeeze(), cmap='pink', alpha=0.4, vmin=0, vmax=1)

    def set_axes(idx):
        my_name, my_image, my_mask, _, _, _ = image_mask_list[data_idx]
        ax.set_title(f'{my_name} s:{slice_index}')
        im0.set_data(my_image[0, :, :, idx].squeeze())
        im1.set_data(my_mask[cls_idx, :, :, idx].squeeze())

    # Function to update the slice
    def update_slice(event):
        nonlocal slice_index
        nonlocal data_idx
        if event.key == 'c':
            if slice_index < my_image.shape[-1] - 1:
                slice_index += 1
                set_axes(slice_index)
                plt.draw()
        elif event.key == 'r':
            if slice_index > 0:
                slice_index -= 1
                set_axes(slice_index)
                plt.draw()
        elif event.key == 'n':
            if data_idx < len(image_mask_list) - 1:
                data_idx += 1
                slice_index = 0
                set_axes(slice_index)
                plt.draw()
        elif event.key == 'p':
            if data_idx > 0:
                data_idx -= 1
                slice_index = 0
                set_axes(slice_index)
                plt.draw()

    # Connect the key press event to the update function
    fig.canvas.mpl_connect('key_press_event', update_slice)

    # Show the plot
    plt.show()


def compare_ours_theirs(data_index, data_path, prefix, m3d_path):
    """
        Compare the image and mask of our pre-processed data with the one from the M3D dataset
    """
    image_mask_list = load_all_image_mask(data_index, data_path, prefix, m3d_path)
    print('loaded', len(image_mask_list), 'images and masks')
    my_name, my_image, my_mask, their_name, their_image, their_mask = image_mask_list[0]

    print('my_image', my_image.shape, 'my_mask', my_mask.shape)
    print('their_image', their_image.shape, 'their_mask', their_mask.shape)
    print('my_mask[0].sum()', my_mask[0].sum())
    print('their_mask[0].sum()', their_mask[0].sum())

    diff_image = my_image - their_image
    diff_mask = np.abs(my_mask - their_mask)

    fig, ax = plt.subplots(2, 3)
    slice_index = 0
    data_idx = 0

    ax[0, 0].set_title(f'{my_name} s:{slice_index}')
    ax[0, 1].set_title(f'{their_name} s:{slice_index}')
    ax[0, 2].set_title('Difference')
    im00 = ax[0, 0].imshow(my_image[0, :, :, slice_index].squeeze(), cmap='gray')
    im01 = ax[0, 1].imshow(their_image[0, :, :, slice_index].squeeze(), cmap='gray')
    im02 = ax[0, 2].imshow(diff_image[0, :, :, slice_index].squeeze(), cmap='gray',
                           vmin=min(np.min(my_image), np.min(their_image)),
                           vmax=max(np.max(my_image), np.max(their_image)))
    im10 = ax[1, 0].imshow(
        my_mask[0, :, :, slice_index].squeeze(), cmap='gray', vmin=0, vmax=1)
    im11 = ax[1, 1].imshow(
        their_mask[0, :, :, slice_index].squeeze(), cmap='gray', vmin=0, vmax=1)
    im12 = ax[1, 2].imshow(
        diff_mask[0, :, :, slice_index].squeeze(), cmap='gray', vmin=0, vmax=1)

    def set_axes(idx):
        my_name, my_image, my_mask, their_name, their_image, their_mask = image_mask_list[data_idx]
        ax[0, 0].set_title(f'{my_name} s:{slice_index}')
        ax[0, 1].set_title(f'{their_name} s:{slice_index}')
        im00.set_data(my_image[0, :, :, idx].squeeze())
        im01.set_data(their_image[0, :, :, idx].squeeze())
        im02.set_data(diff_image[0, :, :, idx].squeeze())
        im10.set_data(my_mask[0, :, :, idx].squeeze())
        im11.set_data(their_mask[0, :, :, idx].squeeze())
        im12.set_data(diff_mask[0, :, :, idx].squeeze())

    # Function to update the slice
    def update_slice(event):
        nonlocal slice_index
        nonlocal data_idx
        if event.key == 'c':
            if slice_index < my_image.shape[-1] - 1:
                slice_index += 1
                set_axes(slice_index)
                plt.draw()
        elif event.key == 'r':
            if slice_index > 0:
                slice_index -= 1
                set_axes(slice_index)
                plt.draw()
        elif event.key == 'n':
            if data_idx < len(image_mask_list) - 1:
                data_idx += 1
                set_axes(slice_index)
                plt.draw()
        elif event.key == 'p':
            if data_idx > 0:
                data_idx -= 1
                set_axes(slice_index)
                plt.draw()

    # Connect the key press event to the update function
    fig.canvas.mpl_connect('key_press_event', update_slice)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_index', type=int, default=None)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='ct_')
    parser.add_argument('--cls_idx', type=int, default=0)
    parser.add_argument('--m3d_path', type=str, default=None)
    args = parser.parse_args()
    if args.m3d_path is not None:
        compare_ours_theirs(args.data_index, args.data_path, args.prefix, args.m3d_path)
    else:
        show_image_mask(args.data_index, args.data_path, args.prefix, args.cls_idx)