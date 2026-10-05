import os
import shutil
import requests
import zipfile
from pathlib import Path


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def download_movielens_1m(dest: str = "data/raw") -> str:
    """Download and extract MovieLens 1M dataset into dest/ml-1m.

    Returns the path to the extracted folder.
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    dest_dir = Path(dest)
    raw_dir = dest_dir / "ml-1m"
    _ensure_dir(dest_dir)

    zip_path = dest_dir / "ml-1m.zip"

    # Download
    print(f"Fetching {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # Extract
    print(f"Extracting to {raw_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

    # MovieLens 1M zip extracts to ml-1m directory
    extracted = dest_dir / "ml-1m"
    if not extracted.exists():
        # Some environments may extract to ml-1m/; if not, try ml-1m folder name
        extracted = dest_dir / "ml-1m"

    # Clean up zip file to save space
    try:
        zip_path.unlink()
    except OSError:
        pass

    print(f"MovieLens 1M ready at: {extracted}")
    return str(extracted)
