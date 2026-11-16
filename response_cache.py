"""
response_cache.py - ONE job: save and load model replies on disk.

Why: the cloud API has limits and runs can stop half way. With the cache a
re-run asks the model only for prompts it has not seen. The cache key contains
the model, temperature, seed, RUN_TAG and the prompt, so changing any of them
gives a fresh call.
"""
import hashlib
import json
import os

from config import (CACHE_DIR, NUM_PREDICT, OLLAMA_MODEL, RUN_TAG, SEED,
                    TEMPERATURE, THINK_MODE)


def make_cache_key(prompt, model=OLLAMA_MODEL, temperature=TEMPERATURE, seed=SEED,
                   run_tag=RUN_TAG, think_mode=THINK_MODE, num_predict=NUM_PREDICT):
    """
    Description: A short unique name for one (settings + prompt) combination.
    Inputs: prompt + all settings (defaults come from config globals).
    Outputs: hex str.
    Dependencies: hashlib, json.
    Utilities: load_cached_reply(), save_cached_reply().
    """
    payload = json.dumps([model, temperature, seed, run_tag, think_mode, num_predict, prompt])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_path(key, cache_dir=None):
    """Description: File path for a key. Inputs: key, cache_dir. Outputs: str.
    Dependencies: os.path. Utilities: load_cached_reply(), save_cached_reply()."""
    return os.path.join(cache_dir or CACHE_DIR, key + ".json")


def load_cached_reply(key, cache_dir=None):
    """
    Description: Return the saved reply (dict) or None if there is none / it is broken.
    Inputs: key, cache_dir. Outputs: dict or None.
    Dependencies: _cache_path(), json. Utilities: OllamaClient.chat().
    """
    path = _cache_path(key, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_cached_reply(key, reply, cache_dir=None):
    """
    Description: Save a reply. Written to a temp file first, so a crash never leaves
        a half-written file.
    Inputs: key, reply (dict), cache_dir. Outputs: none (writes a file).
    Dependencies: _cache_path(), json, os. Utilities: OllamaClient.chat().
    """
    os.makedirs(cache_dir or CACHE_DIR, exist_ok=True)
    path = _cache_path(key, cache_dir)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(reply, f)
    os.replace(tmp_path, path)
