"""
download_cora.py - ONE job: download the three Cora files (a few MB) from Hugging Face
into cora_data/cora/. Run once:   python download_cora.py
"""
from huggingface_hub import hf_hub_download

REPO_ID = "Graph-COM/Text-Attributed-Graphs"
REPO_TYPE = "dataset"
LOCAL_DIR = "cora_data"
FILES = ["cora/processed_data.pt", "cora/raw_texts.pt", "cora/categories.csv"]


def download_one(filename):
    """Description: Download one file. Inputs: filename (path inside the repo).
    Outputs: local path (str). Dependencies: huggingface_hub. Utilities: main()."""
    path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                           filename=filename, local_dir=LOCAL_DIR)
    print("Saved:", path)
    return path


def main():
    """Description: Download all files in FILES. Inputs: none. Outputs: prints paths."""
    for name in FILES:
        download_one(name)


if __name__ == "__main__":
    main()
