import json
import re
import requests
from config import *


def _headers():
    h = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        h["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return h


def _short(text, n=SNIPPET_CHARS):
    return text.replace("\n", " ")[:n]


def format_neighbors(neighbor_ids, texts):
    if not neighbor_ids:
        return "(no neighbors given)"
    return "\n".join(f"[Node {n}] {_short(texts[n])}" for n in neighbor_ids)


def construct_prompt(node_id, texts, neighbor_ids):
    return f"""You classify research papers into one of: {", ".join(CLASS_NAMES)}.

Target paper [Node {node_id}]:
{_short(texts[node_id], 600)}

Neighbor papers (cited by or citing the target):
{format_neighbors(neighbor_ids, texts)}

Reply ONLY with JSON in this format:
{{"label": "<one class>",
  "scores": {{"<class>": <probability>, ...for all classes, sum to 1}},
  "explanation": "<reasoning; cite neighbors like 'Node 123' if they influenced you>"}}"""


def call_ollama(prompt):
    body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": TEMPERATURE, "seed": SEED},
    }
    r = requests.post(
        OLLAMA_URL, headers=_headers(), json=body, timeout=REQUEST_TIMEOUT
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_response(raw):
    """Returns (label, scores, explanation). Bad output -> (None, {}, '')."""
    try:
        d = json.loads(raw)
        label = d.get("label")
        if label not in CLASS_NAMES:
            label = None
        return label, d.get("scores", {}), str(d.get("explanation", ""))
    except json.JSONDecodeError:
        return None, {}, ""


def get_label_prob(scores, label):
    try:
        return float(scores.get(label, 0.0))
    except (TypeError, ValueError):
        return 0.0


def run_inference(prompt):
    return parse_response(call_ollama(prompt))
