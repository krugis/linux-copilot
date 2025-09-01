import os
import requests
import shutil

# URLs for log datasets from Loghub
LOG_DATA_URLS = {
    "Linux": "https://zenodo.org/records/3227177/files/Linux.log.tar.gz",
    "OpenSSH": "https://zenodo.org/records/3227177/files/OpenSSH.log.tar.gz",
    # Add other logs from Loghub as needed
}

def download_file(url, dest_folder):
    """Downloads a file with a progress bar."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_folder, filename)
    
    print(f"Downloading {filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print(f"Successfully downloaded {filename} to {dest_folder}")
    except requests.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")

if __name__ == "__main__":
    OUTPUT_DIR = 'data/cpt/raw/log_data'
    
    for name, url in LOG_DATA_URLS.items():
        print(f"\nProcessing dataset: {name}")
        download_file(url, OUTPUT_DIR)
        
    print("\nLog data download complete.")