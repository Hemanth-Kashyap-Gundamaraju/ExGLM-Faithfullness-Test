"""
prompts.py - ONE job: build the prompt text (copy of the ExGLM paper, Table 11).
"""
from config import HOPS, PROMPT_CLASS_ORDER, PROMPT_STYLE, TITLE_CHARS

TASK_INTRO = ("Classify the article according to its topic into one of the following "
              "categories: [{classes}]. Node represents academic paper with a specific "
              "topic, link represents a citation between the two papers.")

ASK_LINES = {
    "A": "Please answer classification result in the first line.\n"
         "Then provide accurate and critical explanation.",
    "B": "Please answer classification result in the first line.\n"
         "Then provide accurate and critical explanation, naming by title "
         "the connected articles that influenced your decision.",
}


def prompt_title(titles, node_id):
    """
    Description: The title exactly as the model will see it (cut to TITLE_CHARS).
    Inputs: titles (list of str), node_id. Reads global TITLE_CHARS.
    Outputs: str.
    Dependencies: none.
    Utilities: build_prompt(), format_neighbors(), main.py (citation matching).
    """
    return titles[node_id][:TITLE_CHARS]


def quote(text):
    """Description: Put double quotes around text. Inputs: text. Outputs: str.
    Dependencies: none. Utilities: build_prompt(), format_neighbors()."""
    return f'"{text}"'


def format_neighbors(neighbor_ids, titles):
    """Description: Neighbor titles as a quoted, comma-separated list.
    Inputs: neighbor_ids, titles. Outputs: str.
    Dependencies: prompt_title(), quote(). Utilities: build_prompt()."""
    return ", ".join(quote(prompt_title(titles, n)) for n in neighbor_ids)


def build_prompt(node_id, titles, neighbor_ids, style=PROMPT_STYLE, hops=HOPS):
    """
    Description: Full prompt for one target node and the neighbors that are shown.
    Inputs: node_id, titles, neighbor_ids, style, hops.
        Reads globals PROMPT_CLASS_ORDER, PROMPT_STYLE, HOPS.
    Outputs: prompt str.
    Dependencies: prompt_title(), quote(), format_neighbors().
    Utilities: main.py, smoke_test.py.
    """
    intro = TASK_INTRO.format(classes=", ".join(PROMPT_CLASS_ORDER))
    me = quote(prompt_title(titles, node_id))
    if neighbor_ids:
        link = f"{me} is connected with {format_neighbors(neighbor_ids, titles)} within {hops} hops."
    else:
        link = f"{me} has no known connections."
    return f"{intro}\n{link} Which category should {me} be classified as?\n{ASK_LINES[style]}"
