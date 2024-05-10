import zipfile
import os


def unzip_file(zip_path: str, extract_to=None):
    print(f"Extracting {zip_path}")
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)  # Extract all the contents into the directory
        print(f"Files extracted to {extract_to}")
