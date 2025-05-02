"""Download the SpliceAI dataset from Google Cloud Storage."""
#!/usr/bin/env python3
import requests
import sys
import os

def download_file(url, local_path, chunk_size=8192):
    """
    Download a file from `url` to `local_path` in streaming chunks.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = downloaded * 100 / total if total else 0
                    sys.stdout.write(f"\rDownloading {local_path}... {percent:.1f}%")
                    sys.stdout.flush()
    print(f"\nSaved to {local_path}")

def main():
    url = "https://storage.googleapis.com/splice-datasets/splice_dataset.tar.gz"
    local_filename = os.path.basename(url)

    print(f"Fetching:\n  {url}\n->\n  {local_filename}\n")
    download_file(url, local_filename)

if __name__ == "__main__":
    main()
