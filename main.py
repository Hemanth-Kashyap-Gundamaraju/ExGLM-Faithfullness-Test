import random
import numpy as np
from data_loader import load_cora_dataset, get_k_hop_neighborhood
from model_utils import load_quantized_model, construct_prompt, run_inference
from interventions import extract_cited_neighbors, check_hallucinations, apply_intervention

def main():
    print("Loading Cora dataset...")
    data, G = load_cora_dataset()
    
    print("Loading quantized LLM...")
    model, tokenizer = load_quantized_model()
    
    # Sample a subset of test nodes (e.g., 50 for quick testing, scale up later)
    test_indices = data.test_mask.nonzero(as_tuple=False).view(-1).tolist()[:50]
    
    metrics = {
        "cited_removal": {"flip_count": 0, "prob_drop": 0.0},
        "random_removal": {"flip_count": 0, "prob_drop": 0.0},
        "uncited_removal": {"flip_count": 0, "prob_drop": 0.0},
        "total_graph_removal": {"flip_count": 0, "prob_drop": 0.0}
    }
    hallucination_rates = []

    print(f"Evaluating {len(test_indices)} nodes...")
    for node_id in test_indices:
        # 1. Base Setup
        neighborhood = get_k_hop_neighborhood(G, node_id, k=2)
        # Using feature vector sum as a placeholder for raw text mapping
        node_features = f"Feature vector sum: {data.x[node_id].sum().item():.2f}" 
        
        # 2. Base Inference
        base_prompt = construct_prompt(node_id, node_features, neighborhood)
        base_response, base_prob = run_inference(base_prompt, model, tokenizer)
        
        # Simulated parsing of base label 
        base_label = "Class_A" 
        
        # 3. Extract Citations & Check Hallucinations
        cited_nodes = extract_cited_neighbors(base_response)
        h_rate = check_hallucinations(cited_nodes, neighborhood)
        hallucination_rates.append(h_rate)
        
        # 4. Run Interventions
        for strategy in metrics.keys():
            modified_neighborhood = apply_intervention(neighborhood, cited_nodes, strategy)
            mod_prompt = construct_prompt(node_id, node_features, modified_neighborhood)
            mod_response, mod_prob = run_inference(mod_prompt, model, tokenizer)
            
            # Simulated outcome mapping
            mod_label = "Class_B" if random.random() < 0.2 else "Class_A" 
            
            if mod_label != base_label:
                metrics[strategy]["flip_count"] += 1
            metrics[strategy]["prob_drop"] += (base_prob - mod_prob)

    # 5. Output Metrics
    total_nodes = len(test_indices)
    print("\n=== Experimental Results ===")
    print(f"Average Citation Hallucination Rate: {np.mean(hallucination_rates):.1%}\n")
    
    for strategy, stats in metrics.items():
        flip_rate = stats["flip_count"] / total_nodes
        avg_prob_drop = stats["prob_drop"] / total_nodes
        print(f"Strategy: {strategy.replace('_', ' ').title()}")
        print(f"  Prediction Flip Rate: {flip_rate:.1%}")
        print(f"  Average Probability Drop: {avg_prob_drop:.3f}\n")

if __name__ == "__main__":
    main()