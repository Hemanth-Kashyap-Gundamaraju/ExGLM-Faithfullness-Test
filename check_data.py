"""
check_data.py - ONE job: prove the dataset is right (counts, labels, titles,
neighbor sizes, a few samples). Run once after download_cora.py:  python check_data.py
"""
import random
import statistics as st

import torch

from config import CLASS_NAMES, HOPS, MIN_QUOTE_CHARS, SEED
from data_loader import all_test_nodes, get_k_hop_neighborhood, load_cora_with_text

SAMPLE_NODES = 3
TITLE_PREVIEW = 80


def clean_names(data):
    """Description: label names from the file, lowercase. Inputs: data. Outputs: list of str.
    Dependencies: none. Utilities: check_labels(), show_samples()."""
    return [str(x).strip().lower().replace("_", " ") for x in data.label_texts]


def check_counts(data, titles, graph):
    """Description: Print node/edge/test counts. Inputs: data, titles, graph. Outputs: prints."""
    print("== counts ==")
    print("nodes:", data.num_nodes, "| titles:", len(titles))
    print("undirected edges:", graph.number_of_edges())
    ids = set(all_test_nodes(data))
    print("test nodes:", len(ids), "| test_mask matches test_id:", ids == {int(i) for i in data.test_id})


def check_labels(data):
    """Description: Print class counts and check the order matches config.CLASS_NAMES.
    Inputs: data. Reads CLASS_NAMES. Outputs: prints."""
    print("\n== labels ==")
    names = clean_names(data)
    print("same order as CLASS_NAMES in config.py:", names == CLASS_NAMES)
    all_c = torch.bincount(data.y, minlength=len(names)).tolist()
    test_c = torch.bincount(data.y[data.test_mask], minlength=len(names)).tolist()
    for i, n in enumerate(names):
        print(f"  {i} {n:24s} all={all_c[i]:4d}  test={test_c[i]:4d}")


def check_titles(titles):
    """Description: Print title problems after cleaning. Inputs: titles. Reads MIN_QUOTE_CHARS.
    Outputs: prints."""
    print("\n== cleaned titles ==")
    lengths = [len(t) for t in titles]
    short = [i for i, t in enumerate(titles) if len(t) < MIN_QUOTE_CHARS]
    print("empty:", sum(1 for t in titles if not t), "| starting with 'Title:':",
          sum(t.lower().startswith("title:") for t in titles))
    print(f"shorter than {MIN_QUOTE_CHARS} chars (cannot be matched in citations):", len(short))
    print("duplicates:", len(titles) - len(set(titles)))
    print("length min/mean/max:", min(lengths), round(st.mean(lengths)), max(lengths))


def check_neighborhoods(data, graph):
    """Description: Print neighbor-count statistics for test nodes. Inputs: data, graph.
    Reads HOPS. Outputs: prints."""
    print(f"\n== neighbors of test nodes (1-hop and {HOPS}-hop) ==")
    nodes = all_test_nodes(data)
    one = [graph.degree(n) for n in nodes]
    two = [len(get_k_hop_neighborhood(graph, n, HOPS)) for n in nodes]
    for name, vals in (("1-hop", one), (f"{HOPS}-hop", two)):
        print(f"{name}: min={min(vals)} median={st.median(vals)} mean={st.mean(vals):.1f} "
              f"max={max(vals)} | >10: {sum(v > 10 for v in vals)} | >20: {sum(v > 20 for v in vals)}")


def show_samples(data, titles, graph):
    """Description: Print a few test nodes with neighbors, to check by eye.
    Inputs: data, titles, graph. Reads SEED. Outputs: prints."""
    print("\n== samples (title vs label) ==")
    names = clean_names(data)
    for n in random.Random(SEED).sample(all_test_nodes(data), SAMPLE_NODES):
        print(f"\nNode {n} | label: {names[int(data.y[n])]}\n  title: {titles[n][:TITLE_PREVIEW]}")
        for m in list(graph.neighbors(n))[:3]:
            print("   neighbor:", titles[m][:TITLE_PREVIEW], "|", names[int(data.y[m])])


def main():
    """Description: Run all checks. Inputs: none. Outputs: prints."""
    data, titles, graph = load_cora_with_text()
    check_counts(data, titles, graph)
    check_labels(data)
    check_titles(titles)
    check_neighborhoods(data, graph)
    show_samples(data, titles, graph)


if __name__ == "__main__":
    main()
