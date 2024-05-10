import os
from adapt_med_seg.utils.download_file import download_file
from adapt_med_seg.utils.unzip_file import unzip_file
from constants import DATASETS_DIR


def download_m3d_seg_dataset(padded_dataset_number: str) -> None:
    url = f"https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg/resolve/main/M3D_Seg/{padded_dataset_number}.zip?download=true"

    # local_filename = m3d_dataset_dict[padded_dataset_number]
    local_filename = padded_dataset_number
    save_path = os.path.join(DATASETS_DIR, f"{local_filename}.zip")

    if os.path.exists(save_path) or os.path.exists(
        os.path.join(DATASETS_DIR, local_filename)
    ):
        print(f"File already exists as {local_filename}")
        return
    download_file(url, save_path)
    unzip_file(save_path)
    # Delete zip file after unzipping
    os.remove(save_path)

    print(f"File downloaded and saved as {local_filename}")
