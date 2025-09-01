import os
from git import Repo, GitCommandError

# A list of popular and well-structured dotfiles and script repositories
# Sources: Awesome Dotfiles, GitHub Topics
REPOS_TO_CLONE =

def clone_repo(repo_url, base_dir):
    """Clones a single git repository into the specified directory."""
    # Create a directory name from the repo URL
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    clone_path = os.path.join(base_dir, repo_name)
    
    if os.path.exists(clone_path):
        print(f"Repository {repo_name} already exists. Skipping.")
        return

    print(f"Cloning {repo_url} into {clone_path}...")
    try:
        Repo.clone_from(repo_url, clone_path)
        print(f"Successfully cloned {repo_name}.")
    except GitCommandError as e:
        print(f"Error cloning {repo_name}: {e}")

if __name__ == "__main__":
    OUTPUT_DIR = 'data/cpt/raw/github_repos'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for repo_url in REPOS_TO_CLONE:
        clone_repo(repo_url, OUTPUT_DIR)
        
    print("\nFinished cloning all specified repositories.")