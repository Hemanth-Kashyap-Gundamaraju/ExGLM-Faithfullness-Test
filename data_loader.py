"""
data_loader.py - ONE job: load Cora (text + labels + edges) and answer
questions about the graph (neighbors, test nodes).
"""
import random

from config import (ABSTRACT_PREFIX, DATA_PATH, FALLBACK_CHARS, HOPS,
                    MAX_NEIGHBORS, SEED, TITLE_PREFIX)


# ---------- reading the file (I/O) ----------
def read_data_file(path=DATA_PATH):
    """
    Description: Load the Hugging Face Cora file (a PyG Data object).
    Inputs: path (str). Reads global DATA_PATH as default.
    Outputs: the Data object.
    Dependencies: torch.load (imported inside, so tests do not need torch).
    Utilities: called by load_cora_with_text().
    """
    import torch
    return torch.load(path, weights_only=False)


def build_graph(data):
    """
    Description: Turn the Data object into an undirected networkx graph.
    Inputs: data (PyG Data).
    Outputs: networkx Graph.
    Dependencies: torch_geometric.utils.to_networkx.
    Utilities: called by load_cora_with_text().
    """
    from torch_geometric.utils import to_networkx
    return to_networkx(data, to_undirected=True, remove_self_loops=True)


# ---------- titles (pure logic) ----------
def squash_spaces(text):
    """Description: Make all runs of spaces/tabs/newlines a single space.
    Inputs: text. Outputs: cleaned str. Dependencies: none. Utilities: strip_prefix()."""
    return " ".join(str(text).split())


def strip_prefix(text, prefix):
    """Description: Remove a leading word such as 'Title:' (any capital letters).
    Inputs: text, prefix. Outputs: str without the prefix.
    Dependencies: squash_spaces(). Utilities: one_title()."""
    text = squash_spaces(text)
    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].strip()
    return text


def safe_quotes(text):
    """Description: Replace double quotes inside a title, so quote-finding stays correct.
    Inputs: text. Outputs: str. Dependencies: none. Utilities: one_title()."""
    return text.replace('"', "'")


def one_title(data, node_id):
    """
    Description: Clean title of one node. Empty title -> start of the abstract
        -> 'Paper <id>'.
    Inputs: data, node_id. Reads globals TITLE_PREFIX, ABSTRACT_PREFIX, FALLBACK_CHARS.
    Outputs: str.
    Dependencies: strip_prefix(), safe_quotes().
    Utilities: clean_titles().
    """
    title = strip_prefix(data.title[node_id], TITLE_PREFIX)
    if title:
        return safe_quotes(title)
    abstract = strip_prefix(data.abs[node_id], ABSTRACT_PREFIX)
    if abstract:
        return safe_quotes(abstract[:FALLBACK_CHARS])
    return f"Paper {node_id}"


def clean_titles(data):
    """Description: Clean titles of all nodes. Inputs: data. Outputs: list of str.
    Dependencies: one_title(). Utilities: load_cora_with_text()."""
    return [one_title(data, i) for i in range(data.num_nodes)]


def load_cora_with_text(path=DATA_PATH):
    """
    Description: Load everything the experiment needs.
    Inputs: path.
    Outputs: (data, titles, graph).
    Dependencies: read_data_file(), clean_titles(), build_graph().
    Utilities: called by main.py, smoke_test.py, check_data.py.
    """
    data = read_data_file(path)
    return data, clean_titles(data), build_graph(data)


# ---------- neighbors (pure logic) ----------
def get_k_hop_neighborhood(graph, node_id, k=HOPS):
    """
    Description: All nodes within k hops of node_id (the node itself excluded).
    Inputs: graph, node_id, k. Reads global HOPS as default.
    Outputs: sorted list of node ids.
    Dependencies: graph.neighbors.
    Utilities: main.py, check_data.py.
    """
    seen, frontier = {node_id}, {node_id}
    for _ in range(k):
        frontier = {m for n in frontier for m in graph.neighbors(n)} - seen
        seen |= frontier
    seen.discard(node_id)
    return sorted(seen)


def cap_neighbors(graph, node_id, neighborhood, max_n=MAX_NEIGHBORS):
    """
    Description: Keep at most max_n neighbors: 1-hop first, then 2-hop.
        Order is random but fixed by SEED + node_id, so runs repeat exactly.
    Inputs: graph, node_id, neighborhood, max_n. Reads globals SEED, MAX_NEIGHBORS.
    Outputs: list of node ids (the neighbors SHOWN to the model).
    Dependencies: random.Random.
    Utilities: main.py.
    """
    one_hop = sorted(graph.neighbors(node_id))
    one_hop_set = set(one_hop)
    rest = [n for n in neighborhood if n not in one_hop_set]
    rng = random.Random(SEED + node_id)
    rng.shuffle(one_hop)
    rng.shuffle(rest)
    return (one_hop + rest)[:max_n]


def all_test_nodes(data):
    """Description: Sorted ids of all test nodes. Inputs: data. Outputs: list of int.
    Dependencies: none. Utilities: get_test_nodes()."""
    return sorted(data.test_mask.nonzero(as_tuple=False).view(-1).tolist())


def get_test_nodes(data, n):
    """
    Description: First choice = all test nodes. If n is smaller, take a random
        subset with a fixed seed (not the first n, which could be biased).
    Inputs: data, n. Reads global SEED.
    Outputs: sorted list of node ids.
    Dependencies: all_test_nodes().
    Utilities: main.py.
    """
    nodes = all_test_nodes(data)
    if n >= len(nodes):
        return nodes
    return sorted(random.Random(SEED).sample(nodes, n))
