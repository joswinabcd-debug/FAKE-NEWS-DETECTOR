"""
Dataset Downloader Utility for TRUTHSCAN AI.
Downloads the authentic ISOT Fake and Real News Dataset files if they are not already placed in data/.
"""

import os
import sys
import urllib.request
import config

DATA_SOURCES = {
    "Fake.csv": "https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/main/data/Fake.csv",
    "True.csv": "https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/main/data/True.csv"
}


def download_file(url: str, dest_path: str, progress_callback=None) -> None:
    """Downloads a file with real-time percentage progress."""
    print(f"Downloading from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                f.write(buffer)
                
                percent = (downloaded / total_size * 100) if total_size > 0 else 0
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                
                status_str = f"Downloaded {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)"
                sys.stdout.write(f"\r{status_str}")
                sys.stdout.flush()
                
                if progress_callback:
                    progress_callback(percent, status_str)
                    
        print("\nDownload complete.")


def ensure_dataset(progress_callback=None) -> bool:
    """
    Checks if Fake.csv and True.csv exist. If not, downloads them.
    Returns True if dataset files are ready.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    all_present = True
    
    for filename, url in DATA_SOURCES.items():
        dest = os.path.join(config.DATA_DIR, filename)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            all_present = False
            print(f"File {dest} missing. Starting download...")
            download_file(url, dest, progress_callback)
            
    return True


if __name__ == "__main__":
    ensure_dataset()
