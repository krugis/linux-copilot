import os
import subprocess
import requests
import tarfile
import shutil

# The URL for the man-pages archive from kernel.org
MAN_PAGES_URL = "https://www.kernel.org/doc/man-pages/download.html"
# A more direct link might be needed if the above page changes, find the latest.tar.gz
# For example: https://www.kernel.org/pub/linux/docs/man-pages/man-pages-6.05.tar.gz
# Let's find it dynamically
def find_latest_man_pages_url():
    """Finds the latest man-pages tarball URL from the main download page."""
    print("Finding the latest man-pages URL...")
    response = requests.get(MAN_PAGES_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    for a_tag in soup.find_all('a', href=True):
        if 'man-pages-' in a_tag['href'] and a_tag['href'].endswith('.tar.gz'):
            # Construct the full URL
            return f"https://www.kernel.org{a_tag['href']}" if a_tag['href'].startswith('/pub/') else a_tag['href']
    raise RuntimeError("Could not find a download link for man-pages.")

def download_and_extract(url, dest_path):
    """Downloads and extracts a tar.gz file."""
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    filename = url.split('/')[-1]
    tar_path = os.path.join(dest_path, filename)

    print(f"Downloading {filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(tar_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    print(f"Extracting {filename}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=dest_path)
    
    # Find the extracted directory name
    extracted_dir_name = tar.getnames().split('/')
    return os.path.join(dest_path, extracted_dir_name)

def convert_man_to_text(source_dir, output_dir):
    """Converts man pages from roff format to plain text using groff."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Converting man pages from {source_dir} to text in {output_dir}...")
    roff_files_found = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check for man page file extensions (e.g.,.1,.2,.3, etc.)
            if file.split('.')[-1].isdigit():
                roff_files_found += 1
                source_path = os.path.join(root, file)
                output_filename = f"{os.path.splitext(file)}.txt"
                output_path = os.path.join(output_dir, output_filename)
                
                # Use groff to convert to plain text [4, 5]
                command =
                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True, errors='ignore')
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Skipping {source_path}: {e}")
    
    print(f"Found and attempted to convert {roff_files_found} man pages.")

if __name__ == "__main__":
    from bs4 import BeautifulSoup # Import here as it's only needed for the dynamic URL finding
    
    RAW_DATA_PATH = 'data/cpt/raw/man_pages'
    TEXT_DATA_PATH = 'data/cpt/processed/man_pages_text'

    # Step 1: Find, download, and extract the man pages
    try:
        latest_url = find_latest_man_pages_url()
        print(f"Found URL: {latest_url}")
        extracted_path = download_and_extract(latest_url, RAW_DATA_PATH)
        
        # Step 2: Convert the extracted man pages to plain text
        convert_man_to_text(extracted_path, TEXT_DATA_PATH)
        print("\nMan page collection and processing complete.")
        print(f"Raw files are in: {RAW_DATA_PATH}")
        print(f"Processed text files are in: {TEXT_DATA_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")