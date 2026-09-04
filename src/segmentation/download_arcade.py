"""
Download and extract the ARCADE dataset (CC0 license, no registration
required) from its Zenodo record: https://zenodo.org/records/10390295

Run:
    python src/segmentation/download_arcade.py
"""
import os
import sys
import time
import zipfile
import urllib.request

URL = "https://zenodo.org/records/10390295/files/arcade.zip?download=1"
DEST_ZIP = "data/arcade.zip"
DEST_DIR = "data/arcade"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
    sys.stdout.write(f"\r  {downloaded / 1e6:.0f} MB / {total_size / 1e6:.0f} MB ({pct:.1f}%)")
    sys.stdout.flush()


def main():
    os.makedirs("data", exist_ok=True)

    max_attempts = 5
    if not os.path.exists(DEST_ZIP):
        for attempt in range(1, max_attempts + 1):
            print(f"Downloading ARCADE dataset (~450 MB) from Zenodo... (attempt {attempt}/{max_attempts})")
            try:
                urllib.request.urlretrieve(URL, DEST_ZIP, reporthook=_progress)
                print("\nDownload complete.")
                break
            except Exception as e:
                print(f"\n  download failed: {e}")
                if os.path.exists(DEST_ZIP):
                    os.remove(DEST_ZIP)  # remove partial file before retrying
                if attempt == max_attempts:
                    raise
                wait = 5 * attempt
                print(f"  retrying in {wait}s...")
                time.sleep(wait)
    else:
        print("Zip already downloaded, skipping.")

    if not os.path.exists(DEST_DIR):
        print("Extracting...")
        with zipfile.ZipFile(DEST_ZIP, "r") as zf:
            zf.extractall(DEST_DIR)
        print(f"Extracted to {DEST_DIR}")
    else:
        print("Already extracted, skipping.")

    print("\nContents of", DEST_DIR, ":")
    for entry in sorted(os.listdir(DEST_DIR))[:10]:
        print(" -", entry)
    print("\nIMPORTANT: check this listing against what dataset.py expects")
    print("(it assumes <DEST_DIR>/syntax/<split>/images and .../annotations).")
    print("If the zip extracted into an extra nested folder, adjust the")
    print("`root` path passed to ArcadeSegmentationDataset accordingly.")


if __name__ == "__main__":
    main()