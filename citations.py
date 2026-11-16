"""
citations.py - ONE job: find which neighbors an explanation cites.
Pure functions (text in, numbers out).

The ExGLM paper's explanations cite neighbors BY TITLE (Tables 11-14), so we
match titles, not "Node 123".
"""
import difflib
import re
from dataclasses import dataclass, field
from typing import List

from config import CLASS_NAMES, MATCH_THRESHOLD, MIN_QUOTE_CHARS

QUOTE_PATTERN = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]')


@dataclass
class CitationResult:
    """Description: valid_cited = shown neighbors the explanation names;
    bad_spans = quoted titles that match no shown neighbor (possible hallucinations);
    n_unmatchable = shown neighbors with a title too short to match."""
    valid_cited: List[int] = field(default_factory=list)
    bad_spans: List[str] = field(default_factory=list)
    n_unmatchable: int = 0


def norm_text(text):
    """Description: lowercase, keep only letters/digits/spaces. Inputs: text.
    Outputs: str. Dependencies: re. Utilities: all functions below."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def is_same_title(span, title):
    """
    Description: Is a quoted span a (maybe shortened) copy of a title?
    Inputs: span, title (both already normalized). Reads global MATCH_THRESHOLD.
    Outputs: bool. Dependencies: difflib. Utilities: find_citations().
    """
    if span in title or title in span:
        return True
    return difflib.SequenceMatcher(None, span, title).ratio() >= MATCH_THRESHOLD


def quoted_spans(explanation):
    """
    Description: Quoted pieces of the explanation that look like titles
        (long enough, and not just a class name).
    Inputs: explanation. Reads globals MIN_QUOTE_CHARS, CLASS_NAMES.
    Outputs: list of normalized spans (no duplicates).
    Dependencies: norm_text(). Utilities: find_citations().
    """
    class_set = {norm_text(c) for c in CLASS_NAMES}
    spans = {norm_text(s) for s in QUOTE_PATTERN.findall(explanation)}
    return sorted(s for s in spans if len(s) >= MIN_QUOTE_CHARS and s not in class_set)


def matchable_titles(shown, title_of):
    """Description: {node: normalized title} for shown neighbors whose title is long
    enough to match. Inputs: shown (ids), title_of (function id->title).
    Outputs: dict. Dependencies: norm_text(). Utilities: find_citations()."""
    pairs = {n: norm_text(title_of(n)) for n in shown}
    return {n: t for n, t in pairs.items() if len(t) >= MIN_QUOTE_CHARS}


def find_citations(explanation, shown, target_title, title_of):
    """
    Description: Which shown neighbors does the explanation cite, and which quoted
        titles match nothing?
    Inputs: explanation, shown (neighbor ids in the prompt), target_title, title_of.
    Outputs: CitationResult.
    Dependencies: quoted_spans(), matchable_titles(), is_same_title(), norm_text().
    Utilities: main.py. Note: bad_spans is an UPPER bound, because a quoted phrase that
        is not a title (for example a topic name) is also counted as bad.
    """
    titles = matchable_titles(shown, title_of)
    norm_explanation, target = norm_text(explanation), norm_text(target_title)
    cited = {n for n, t in titles.items() if t in norm_explanation}   # unquoted mentions
    bad = []
    for span in quoted_spans(explanation):
        hits = [n for n, t in titles.items() if is_same_title(span, t)]
        if hits:
            cited.update(hits)
        elif not is_same_title(span, target):
            bad.append(span)
    return CitationResult(sorted(cited), bad, len(list(shown)) - len(titles))


def hallucination_rate(result):
    """Description: share of quoted titles that match no shown neighbor (0 if none).
    Inputs: CitationResult. Outputs: float. Dependencies: none. Utilities: metrics.py/main.py."""
    total = len(result.valid_cited) + len(result.bad_spans)
    return len(result.bad_spans) / total if total else 0.0


def is_explanation_consistent(explanation, label):
    """
    Description: False if the explanation talks about other classes but never
        about its own label (the paper's Table 13 problem).
    Inputs: explanation, label. Reads global CLASS_NAMES.
    Outputs: bool. Dependencies: none. Utilities: main.py.
    """
    text = explanation.lower().replace("-", " ").replace("_", " ")
    others = [c for c in CLASS_NAMES if c != label and c in text]
    return label in text or not others
