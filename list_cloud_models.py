"""
list_cloud_models.py - ONE job: show which models your Ollama cloud key can use.
Copy a model name EXACTLY as printed into OLLAMA_MODEL.   python list_cloud_models.py
"""
import requests

from config import OLLAMA_API_KEY
from ollama_client import build_headers

TAGS_URL = "https://ollama.com/api/tags"
HIGHLIGHT = "qwen"


def fetch_model_names(api_key=OLLAMA_API_KEY):
    """Description: Ask ollama.com for the model list. Inputs: api_key. Outputs: sorted list of names.
    Dependencies: requests, build_headers(). Utilities: main()."""
    response = requests.get(TAGS_URL, headers=build_headers(api_key), timeout=60)
    response.raise_for_status()
    return sorted(m.get("name") or m.get("model") for m in response.json().get("models", []))


def main():
    """Description: Print all models, then the Qwen ones. Inputs: none. Outputs: prints."""
    names = fetch_model_names()
    print(f"{len(names)} models available:")
    print("\n".join("  " + n for n in names))
    print(f"\nModels containing '{HIGHLIGHT}':")
    print("\n".join("  " + n for n in names if HIGHLIGHT in n.lower()) or "  (none)")


if __name__ == "__main__":
    main()
