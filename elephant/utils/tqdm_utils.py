import requests
import tarfile
from tqdm import tqdm


def download_with_progress_bar(url: str, target_file: str):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(target_file, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


def extract_with_progress_bar(tar: str, tar_root_path: str, output_path: str):
    with tarfile.open(tar, "r:gz") as t:
        progress_bar = tqdm(total=len(t.getmembers()))
        for m in t.getmembers():
            progress_bar.update()
            prefix = f"{tar_root_path}/" if tar_root_path else ""
            m.name = m.name.lstrip(prefix).lower()
            t.extract(m, output_path)
        progress_bar.set_description("")
        progress_bar.close()
