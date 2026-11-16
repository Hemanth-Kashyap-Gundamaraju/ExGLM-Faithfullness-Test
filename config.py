"""
config.py - every setting of the project lives here (ALL_CAPS globals).

Most settings can be changed from outside with environment variables, so
run_sweep.py can change one knob at a time without editing any file.
"""
import os


def _env(name, default, cast=str):
    """
    Description: Read one setting from an environment variable, or use the default.
    Inputs: name (str), default (any), cast (function such as int/float/str).
    Outputs: the value, converted with cast.
    Dependencies: os.getenv.
    Utilities: used only in this file.
    """
    return cast(os.getenv(name, default))


# ---------------- Model (Ollama) ----------------
# Cloud API (default). For a local/Colab Ollama use OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_URL = _env("OLLAMA_URL", "https://ollama.com/api/chat")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")          # NEVER write the key in a file
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen3-next:80b")      # CHANGE: copy a name from list_cloud_models.py
THINK_MODE = _env("THINK_MODE", "default")                 # "default" | "off" (think=false) | "low"/"medium"/"high" (gpt-oss)
USE_LOGPROBS = _env("USE_LOGPROBS", "1", int)              # 1 = ask for token probabilities
TOP_LOGPROBS = 20
NUM_PREDICT = _env("NUM_PREDICT", "600", int)              # max new tokens (thinking models need room)
NUM_PREDICT_RERUN = _env("NUM_PREDICT_RERUN", "40", int)   # re-runs after removal only need the label line

# ---------------- Sensitivity-study knobs (from the proposal) ----------------
TEMPERATURE = _env("TEMPERATURE", "0.0", float)            # 0.0 or 0.7
HOPS = _env("HOPS", "2", int)                              # 1 or 2
MAX_NEIGHBORS = _env("MAX_NEIGHBORS", "10", int)           # 5, 10, 20
PROMPT_STYLE = _env("PROMPT_STYLE", "A")                   # A = paper wording, B = also "name articles by title"

# ---------------- Run settings ----------------
SEED = _env("SEED", "42", int)
RUN_TAG = _env("RUN_TAG", "run1")                          # change it to get a fresh sample at T=0.7
NUM_TEST_NODES = _env("NUM_TEST_NODES", "542", int)        # Cora test set has 542 nodes
WORKERS = _env("WORKERS", "4", int)                        # parallel requests
REQUEST_TIMEOUT = 300
MAX_RETRIES = 4
RETRY_SLEEP_SECONDS = 5
RUN_NAME = _env("RUN_NAME", "")                            # if set, results are saved as results/<RUN_NAME>.json

# ---------------- Paths ----------------
DATA_PATH = _env("DATA_PATH", "cora_data/cora/processed_data.pt")
CACHE_DIR = _env("CACHE_DIR", "cache")
RESULTS_DIR = "results"

# ---------------- Classes ----------------
# SAME ORDER AS THE DATA (label 0..6). Used to decide correct vs incorrect.
CLASS_NAMES = ["case based", "genetic algorithms", "neural networks",
               "probabilistic methods", "reinforcement learning",
               "rule learning", "theory"]
# Order shown to the model: same as the ExGLM paper's Table 11.
PROMPT_CLASS_ORDER = ["theory", "reinforcement learning", "genetic algorithms",
                      "neural networks", "probabilistic methods", "case based",
                      "rule learning"]

# ---------------- Text and citation matching ----------------
TITLE_CHARS = 150          # titles are cut to this length in the prompt (paper shows titles only)
TITLE_PREFIX = "Title:"
ABSTRACT_PREFIX = "Abstract:"
FALLBACK_CHARS = 100       # empty title -> first characters of the abstract
MATCH_THRESHOLD = 0.85     # how close a quoted title must be to a real one
MIN_QUOTE_CHARS = 12       # shorter titles/quotes are ignored when matching
MIN_PREFIX_CHARS = 3       # token must have 3+ letters to count for class probability

# ---------------- Interventions ----------------
CONTROL_STRATEGY = "no_change"   # same prompt again: shows how much changes by noise alone
STRATEGIES = ["cited_removal", "random_removal", "uncited_removal",
              "total_graph_removal", CONTROL_STRATEGY]
GROUPS = ("all", "correct", "incorrect", "citers")   # citers = nodes that cited >= 1 shown neighbor
