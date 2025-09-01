import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def scrape_page_text(url):
    """Scrapes all paragraph text from a single page."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # This selector is a common starting point for main content areas.
        # You WILL need to inspect each target site to find the correct selectors.
        content_selectors = ['article', 'main', '.main-content', '#main-content', '.content', 'body']
        content = None
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                break
        
        if not content:
            return ""

        # Extract text from all paragraphs within the content area
        paragraphs = content.find_all('p')
        text = "\n".join()
        return text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""

def save_text_to_file(url, text, base_dir):
    """Saves the scraped text to a file."""
    if not text:
        return
        
    parsed_url = urlparse(url)
    # Create a filename from the URL path
    filename = parsed_url.path.strip('/').replace('/', '_') + '.txt'
    if not filename:
        filename = "index.txt"
        
    domain_dir = os.path.join(base_dir, parsed_url.netloc)
    if not os.path.exists(domain_dir):
        os.makedirs(domain_dir)
        
    filepath = os.path.join(domain_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved content from {url} to {filepath}")

if __name__ == "__main__":
    # List of starting URLs for documentation sites
    # Add more URLs to expand the dataset
    URLS_TO_SCRAPE = [
        "https://docs.docker.com/get-started/",
        "https://kubernetes.io/docs/concepts/overview/",
        "https://manpages.ubuntu.com/manpages/jammy/en/man1/ls.1.html"
    ]
    
    OUTPUT_DIR = 'data/cpt/raw/scraped_docs'

    for url in URLS_TO_SCRAPE:
        print(f"\nScraping {url}...")
        page_text = scrape_page_text(url)
        save_text_to_file(url, page_text, OUTPUT_DIR)
        
    print("\nDocumentation scraping complete.")