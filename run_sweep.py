"""
run_sweep.py - ONE job: run the sensitivity study. It changes ONE setting at a time
from the base setting (so about 9 runs, not hundreds) and saves results/<name>.json
and results/<name>.txt for each run. Finished runs are skipped, so you can stop and
restart.     python run_sweep.py
"""
import os
import subprocess
import sys

from config import OLLAMA_MODEL, RESULTS_DIR, THINK_MODE

SWEEP_NUM_NODES = os.getenv("SWEEP_NUM_NODES", "542")   # set 200 if time is short
OTHER_MODEL = os.getenv("OTHER_MODEL", "CHANGE-ME")   # second model for the size run, e.g. qwen2.5:7b or gpt-oss:120b

BASE = {"OLLAMA_MODEL": OLLAMA_MODEL, "TEMPERATURE": "0.0", "HOPS": "2",
        "MAX_NEIGHBORS": "10", "PROMPT_STYLE": "A", "RUN_TAG": "run1", "THINK_MODE": THINK_MODE,
        "NUM_TEST_NODES": SWEEP_NUM_NODES}

CHANGES = [
    {},                                              # base run
    {"PROMPT_STYLE": "B"},                           # prompt wording
    {"TEMPERATURE": "0.7"},                          # temperature
    {"TEMPERATURE": "0.7", "RUN_TAG": "run2"},       # same again: how much does sampling noise move the numbers?
    {"HOPS": "1"},                                   # 1-hop only
    {"MAX_NEIGHBORS": "5"},                          # fewer neighbors
    {"MAX_NEIGHBORS": "20"},                         # more neighbors
    {"OLLAMA_MODEL": OTHER_MODEL},                   # model size
]


def short_value(value):
    """Description: Make a setting value safe for a file name. Inputs: value (str).
    Outputs: str. Dependencies: none. Utilities: make_run_name()."""
    return value.replace(":", "").replace("/", "").replace(".", "p")


def make_run_name(change):
    """
    Description: File-name-safe name made from the CHANGED settings ('base' if none).
    Inputs: change (dict). Outputs: str. Dependencies: short_value(). Utilities: run_one().
    """
    if not change:
        return "base"
    return "_".join(f"{k.lower()}-{short_value(v)}" for k, v in change.items())


def should_skip(change, results_dir=RESULTS_DIR):
    """Description: True if this run is not ready (placeholder model) or already finished.
    Inputs: change, results_dir. Outputs: bool. Dependencies: os. Utilities: run_one()."""
    if change.get("OLLAMA_MODEL") == "CHANGE-ME":
        print("Skipping the model-size run: set the OTHER_MODEL environment variable first.")
        return True
    path = os.path.join(results_dir, make_run_name(change) + ".json")
    if os.path.exists(path):
        print(f"Already done: {path}")
        return True
    return False


def run_one(change):
    """
    Description: Start main.py in a new process with the base settings + this change.
    Inputs: change (dict). Reads BASE, RESULTS_DIR. Outputs: writes results files.
    Dependencies: subprocess, make_run_name(), should_skip(). Utilities: main().
    """
    if should_skip(change):
        return
    name = make_run_name(change)
    env = {**os.environ, **BASE, **change, "RUN_NAME": name}
    print(f"\n=== Running: {name} ===")
    with open(os.path.join(RESULTS_DIR, name + ".txt"), "w") as out:
        subprocess.run([sys.executable, "main.py"], env=env, stdout=out, check=False)


def main():
    """Description: Run every change in CHANGES. Inputs: none. Outputs: files in results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for change in CHANGES:
        run_one(change)


if __name__ == "__main__":
    main()
