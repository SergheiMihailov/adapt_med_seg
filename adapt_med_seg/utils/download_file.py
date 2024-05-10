import requests
from tqdm import tqdm


def download_file(url: str, local_filename: str) -> None:
    # Stream the download to handle large files without running out of memory
    print(f"Downloading {url} to {local_filename}")
    with requests.get(url, stream=True, timeout=3600) as r:
        r.raise_for_status()  # Checks for a successful download, raises HTTPError for bad status
        with open(local_filename, "wb") as f:
            for chunk in tqdm(
                r.iter_content(chunk_size=8192),
                total=int(r.headers["Content-Length"] / 8192),
            ):
                f.write(chunk)  # Write the content in chunks to the file
