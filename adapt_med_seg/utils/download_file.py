import requests
from tqdm import tqdm
import os

def download_file(url: str, local_filename: str) -> None:

    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    if not os.path.exists(local_filename):
        open(local_filename, 'a').close()
        print(f"File {local_filename} created.")
    else:
        return
    
    print(f"Downloading {url} to {local_filename}")
    with requests.get(url, stream=True, timeout=3600) as r:
        r.raise_for_status()  
        with open(local_filename, "wb") as f:
            for chunk in tqdm(
                r.iter_content(chunk_size=8192),
                total=int(int(r.headers.get("Content-Length", 0)) / 8192),
            ):
                f.write(chunk)  
