"""
main.py - ONE job: run the faithfulness experiment for the settings in config.py
and print / save the report.

Usage:   python main.py
Quick test (3 nodes):   NUM_TEST_NODES=3 python main.py
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from citations import find_citations, hallucination_rate, is_explanation_consistent
from config import (CLASS_NAMES, CONTROL_STRATEGY, HOPS, MAX_NEIGHBORS, NUM_PREDICT_RERUN, NUM_TEST_NODES,
                    OLLAMA_MODEL, PROMPT_STYLE, RESULTS_DIR, RUN_NAME, RUN_TAG, SEED,
                    STRATEGIES, TEMPERATURE, WORKERS)
from data_loader import cap_neighbors, get_k_hop_neighborhood, get_test_nodes, load_cora_with_text
from interventions import apply_intervention
from metrics import MetricsTable, NodeResult, format_report, summarize_run
from ollama_client import FatalOllamaError, OllamaClient
from prompts import build_prompt, prompt_title
from response_parser import parse_reply

PROGRESS_EVERY = 25


@dataclass
class Experiment:
    """Description: Everything shared by all nodes (data, graph, model client)."""
    data: Any
    titles: list
    graph: Any
    client: Any


def seed_offset_for(strategy):
    """Description: The control uses another random seed so it is a fresh sample.
    Inputs: strategy. Reads global CONTROL_STRATEGY. Outputs: int (0 or 1).
    Dependencies: none. Utilities: run_strategy()."""
    return 1 if strategy == CONTROL_STRATEGY else 0


def run_strategy(strategy, node_id, exp, shown, valid_cited, base_label, base_prob):
    """
    Description: Change the neighbor list, ask the model again, compare with the base answer.
    Inputs: strategy, node_id, exp, shown (neighbors in the base prompt), valid_cited,
        base_label, base_prob.
    Outputs: (flipped, prob_drop) or None if the answer was unreadable.
        prob_drop = base_prob - new probability of the ORIGINAL label (None if no probabilities).
    Dependencies: apply_intervention(), build_prompt(), client.chat(), parse_reply().
    Utilities: evaluate_node().
    """
    new_neighbors = apply_intervention(shown, valid_cited, strategy, node_id)
    prompt = build_prompt(node_id, exp.titles, new_neighbors)
    reply = exp.client.chat(prompt, seed_offset=seed_offset_for(strategy), num_predict=NUM_PREDICT_RERUN)
    parsed = parse_reply(reply, prob_label=base_label)
    if parsed.label is None:
        return None
    drop = None
    if base_prob is not None and parsed.prob is not None:
        drop = base_prob - parsed.prob
    return parsed.label != base_label, drop


def evaluate_node(node_id, exp):
    """
    Description: The full test for one node: base answer, citations, 5 interventions.
    Inputs: node_id, exp (Experiment).
    Outputs: NodeResult, or None if the base answer was unreadable / the call failed.
    Dependencies: data_loader functions, build_prompt(), parse_reply(), find_citations(),
        run_strategy().
    Utilities: run_all_nodes().
    """
    try:
        full = get_k_hop_neighborhood(exp.graph, node_id)
        shown = cap_neighbors(exp.graph, node_id, full)
        base = parse_reply(exp.client.chat(build_prompt(node_id, exp.titles, shown)))
        if base.label is None:
            return None
        title_of = lambda n: prompt_title(exp.titles, n)
        citations = find_citations(base.explanation, shown, title_of(node_id), title_of)
        result = NodeResult(
            node_id=node_id, base_label=base.label,
            is_correct=base.label == CLASS_NAMES[int(exp.data.y[node_id])],
            base_prob=base.prob or None, valid_cited=citations.valid_cited,   # 0.0 = first token was not the label
            n_bad_spans=len(citations.bad_spans),
            hallucination_rate=hallucination_rate(citations),
            is_consistent=is_explanation_consistent(base.explanation, base.label),
            n_full_neighborhood=len(full), n_shown=len(shown))
        for strategy in STRATEGIES:
            result.outcomes[strategy] = run_strategy(strategy, node_id, exp, shown,
                                                     citations.valid_cited, base.label, result.base_prob)
        return result
    except FatalOllamaError:
        raise
    except RuntimeError as error:
        print(f"  node {node_id} skipped: {error}")
        return None


def run_all_nodes(exp, node_ids):
    """
    Description: Evaluate all nodes, several at the same time (WORKERS threads).
    Inputs: exp, node_ids. Reads global WORKERS, PROGRESS_EVERY.
    Outputs: list of NodeResult (None for skipped nodes), in node order.
    Dependencies: evaluate_node(). Utilities: main().
    """
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, result in enumerate(pool.map(lambda n: evaluate_node(n, exp), node_ids), 1):
            results.append(result)
            if done % PROGRESS_EVERY == 0 or done == len(node_ids):
                print(f"  {done}/{len(node_ids)} nodes done", flush=True)
    return results


def settings_dict():
    """Description: The settings of this run (saved next to the numbers).
    Inputs: none. Reads config globals. Outputs: dict. Dependencies: none. Utilities: save_results()."""
    return {"model": OLLAMA_MODEL, "temperature": TEMPERATURE, "hops": HOPS,
            "max_neighbors": MAX_NEIGHBORS, "prompt_style": PROMPT_STYLE,
            "num_test_nodes": NUM_TEST_NODES, "seed": SEED, "run_tag": RUN_TAG}


def save_results(table, stats, results_dir=RESULTS_DIR, run_name=RUN_NAME):
    """
    Description: Save settings + run stats + metrics as results/<run_name>.json
        (only if RUN_NAME is set).
    Inputs: table, stats, results_dir, run_name. Outputs: path or None (writes a file).
    Dependencies: json, os, settings_dict(). Utilities: main().
    """
    if not run_name:
        return None
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, run_name + ".json")
    with open(path, "w") as f:
        json.dump({"settings": settings_dict(), "stats": stats, "metrics": table.to_dict()}, f, indent=2)
    return path


def build_experiment():
    """Description: Load data and create the client. Inputs: none. Outputs: Experiment.
    Dependencies: load_cora_with_text(), OllamaClient. Utilities: main()."""
    data, titles, graph = load_cora_with_text()
    return Experiment(data=data, titles=titles, graph=graph, client=OllamaClient())


def main():
    """Description: Run everything and print the report. Inputs: none (config globals).
    Outputs: prints the report; may save a JSON file."""
    print(f"Settings: {settings_dict()}")
    exp = build_experiment()
    node_ids = get_test_nodes(exp.data, NUM_TEST_NODES)
    print(f"Evaluating {len(node_ids)} test nodes...")
    results = run_all_nodes(exp, node_ids)
    good = [r for r in results if r is not None]
    table = MetricsTable()
    for result in good:
        table.add(result)
    stats = summarize_run(good, n_skipped=len(results) - len(good))
    print("\n" + format_report(table, stats))
    path = save_results(table, stats)
    if path:
        print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
