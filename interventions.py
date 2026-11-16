"""
interventions.py - ONE job: change the neighbor list for each test (pure logic).
"""
import random

from config import CONTROL_STRATEGY, SEED


def remove_nodes(neighborhood, to_remove):
    """Description: neighborhood without the nodes in to_remove (order kept).
    Inputs: neighborhood (list), to_remove (iterable). Outputs: list.
    Dependencies: none. Utilities: apply_intervention()."""
    banned = set(to_remove)
    return [n for n in neighborhood if n not in banned]


def pick_random_uncited(neighborhood, valid_cited, node_id):
    """
    Description: k random neighbors that were NOT cited, k = number of cited ones.
        Fixed seed (SEED + node_id), so the same nodes are picked in every run.
    Inputs: neighborhood, valid_cited, node_id. Reads global SEED.
    Outputs: list of node ids.
    Dependencies: random.Random. Utilities: apply_intervention().
    """
    pool = [n for n in neighborhood if n not in set(valid_cited)]
    k = min(len(valid_cited), len(pool))
    return random.Random(SEED + node_id).sample(pool, k)


def apply_intervention(neighborhood, valid_cited, strategy, node_id):
    """
    Description: The neighbor list after one intervention.
        cited_removal        : remove the cited neighbors
        random_removal       : remove the same number of random UNcited neighbors (main baseline)
        uncited_removal      : keep ONLY the cited neighbors (sufficiency test)
        total_graph_removal  : remove all neighbors (text-only baseline)
        no_change            : control, nothing removed
    Inputs: neighborhood (shown ids), valid_cited, strategy (str), node_id. Reads CONTROL_STRATEGY.
    Outputs: list of node ids.
    Dependencies: remove_nodes(), pick_random_uncited().
    Utilities: main.py.
    """
    if strategy == "cited_removal":
        return remove_nodes(neighborhood, valid_cited)
    if strategy == "random_removal":
        return remove_nodes(neighborhood, pick_random_uncited(neighborhood, valid_cited, node_id))
    if strategy == "uncited_removal":
        return [n for n in neighborhood if n in set(valid_cited)]
    if strategy == "total_graph_removal":
        return []
    if strategy == CONTROL_STRATEGY:
        return list(neighborhood)
    raise ValueError(f"Unknown strategy: {strategy}")
