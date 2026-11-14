import re
import random

def extract_cited_neighbors(explanation):
    """Parses the generated explanation to find explicitly cited node IDs."""
    matches = re.findall(r"Node\s(\d+)", explanation, re.IGNORECASE)
    return list(set([int(m) for m in matches]))

def check_hallucinations(cited_nodes, actual_neighborhood):
    """Calculates the percentage of cited nodes missing from the true neighborhood."""
    hallucinated = [n for n in cited_nodes if n not in actual_neighborhood]
    return len(hallucinated) / len(cited_nodes) if cited_nodes else 0.0

def apply_intervention(neighborhood, cited_nodes, strategy):
    """Modifies the target node's neighborhood based on the ablation strategy."""
    neighborhood_set = set(neighborhood)
    cited_set = set(cited_nodes).intersection(neighborhood_set)
    
    if strategy == "cited_removal":
        return list(neighborhood_set - cited_set)
    
    elif strategy == "random_removal":
        k = len(cited_set)
        uncited_nodes = list(neighborhood_set - cited_set)
        if len(uncited_nodes) >= k:
            to_remove = random.sample(uncited_nodes, k)
        else:
            to_remove = uncited_nodes
        return list(neighborhood_set - set(to_remove))
        
    elif strategy == "uncited_removal":
        return list(cited_set) # Sufficiency test
        
    elif strategy == "total_graph_removal":
        return []
        
    return neighborhood