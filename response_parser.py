"""
response_parser.py - ONE job: turn the model's reply text into label, explanation
and class probability. Pure functions (no network, no files).
"""
import math
import re
from dataclasses import dataclass
from typing import Optional

from config import CLASS_NAMES, MIN_PREFIX_CHARS


@dataclass
class ParsedReply:
    """Description: What we keep from one reply.
    label: class name or None if the reply could not be understood.
    explanation: text after the first line. prob: probability of a chosen label or None."""
    label: Optional[str]
    explanation: str
    prob: Optional[float]


def normalize_label_text(text):
    """Description: lowercase, '-' and '_' become spaces, only letters/spaces kept.
    Inputs: text. Outputs: str. Dependencies: re. Utilities: find_label(), class_probability()."""
    text = text.lower().replace("-", " ").replace("_", " ")
    return " ".join(re.sub(r"[^a-z ]", " ", text).split())


def find_label(first_line):
    """
    Description: Which class name is written in the first line? Exactly one must appear.
    Inputs: first_line. Reads global CLASS_NAMES.
    Outputs: class name or None.
    Dependencies: normalize_label_text(). Utilities: split_response().
    """
    line = normalize_label_text(first_line)
    found = [c for c in CLASS_NAMES if c in line]
    return found[0] if len(found) == 1 else None


def split_response(text):
    """
    Description: First non-empty line = label line, the rest = explanation.
    Inputs: text. Outputs: (label or None, explanation str).
    Dependencies: find_label(). Utilities: parse_reply().
    """
    lines = [line for line in (text or "").strip().split("\n") if line.strip()]
    if not lines:
        return None, ""
    return find_label(lines[0]), "\n".join(lines[1:])


def class_start_entry(logprobs):
    """
    Description: The log-probability entry of the token where the class name STARTS in
        the first line. Models often write "**Category: Reinforcement Learning**", so
        the first token is not the class; we look for the first token (3+ letters) that
        is the beginning of some class name. Stops at the end of the first line.
    Inputs: logprobs (list of token entries from the reply, or None). Reads CLASS_NAMES,
        MIN_PREFIX_CHARS.
    Outputs: entry dict or None.
    Dependencies: normalize_label_text().
    Utilities: parse_reply(), smoke_test.py.
    """
    compact_classes = [c.replace(" ", "") for c in CLASS_NAMES]
    seen_text = False
    for entry in logprobs or []:
        token = entry.get("token", "")
        if "\n" in token and seen_text:
            return None
        compact = normalize_label_text(token).replace(" ", "")
        seen_text = seen_text or bool(compact)
        if len(compact) >= MIN_PREFIX_CHARS and any(c.startswith(compact) for c in compact_classes):
            return entry
    return None


def class_probability(entry, label):
    """
    Description: Probability that the token at the class start begins the given class name.
        Adds up all candidate tokens (3+ letters) that are the beginning of the label.
        If the server gave candidates but none matches, the class is less likely than
        the 20th candidate, so we return 0.0. If the server gave no candidates at all
        we return None (unknown) - we never invent a number.
    Inputs: entry (from class_start_entry), label. Reads global MIN_PREFIX_CHARS.
    Outputs: float or None.
    Dependencies: normalize_label_text(), math.
    Utilities: parse_reply().
    """
    if not entry or not label:
        return None
    candidates = list(entry.get("top_logprobs") or [])
    if "token" in entry and "logprob" in entry:
        candidates.append({"token": entry["token"], "logprob": entry["logprob"]})
    if not candidates:
        return None
    seen, total = set(), 0.0
    for alt in candidates:
        token = normalize_label_text(alt.get("token", "")).replace(" ", "")
        key = (alt.get("token"), alt.get("logprob"))
        if key in seen:
            continue
        seen.add(key)
        if len(token) >= MIN_PREFIX_CHARS and label.replace(" ", "").startswith(token):
            total += math.exp(alt["logprob"])
    return total


def parse_reply(reply, prob_label=None):
    """
    Description: Full parsing of one Ollama reply.
    Inputs: reply dict, prob_label (class whose probability we want; default = predicted label).
    Outputs: ParsedReply.
    Dependencies: split_response(), class_start_entry(), class_probability().
    Utilities: main.py.
    """
    content = (reply.get("message") or {}).get("content", "")
    label, explanation = split_response(content)
    prob = class_probability(class_start_entry(reply.get("logprobs")), prob_label or label)
    return ParsedReply(label=label, explanation=explanation, prob=prob)
