import numpy as np


def process_ct_image(image_ndarray):
    """
    Process the CT image.
    Filter out the background (less than the mean value),
    clip the values to the 99.95th percentile of the foreground values,
    and standardize the values by the mean and std of the foreground.
    """
    """
    ct_voxel_ndarray = image_ndarray.copy()
    ct_voxel_ndarray = ct_voxel_ndarray.flatten()
    # for all data
    thred = np.mean(ct_voxel_ndarray)
    voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
    # for foreground data
    upper_bound = np.percentile(voxel_filtered, 99.95)
    lower_bound = np.percentile(voxel_filtered, 00.05)
    mean = np.mean(voxel_filtered)
    std = np.std(voxel_filtered)
    ### transform ###
    image_ndarray = np.clip(image_ndarray, lower_bound, upper_bound)
    image_ndarray = (image_ndarray - mean) / max(std, 1e-8)
    return image_ndarray"""
    return image_ndarray


def process_mr_image(image_ndarray):
    """
    Process the MR image
    """
    # for now this is the same as the CT image
    # TODO: implement a more sophisticated preprocessing method
    #return process_ct_image(image_ndarray)
    return image_ndarray
