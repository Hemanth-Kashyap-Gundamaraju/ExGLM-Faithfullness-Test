"""
summarize_results.py - ONE job: read results/*.json and print one table with a row per run
(paste it into the report).     python summarize_results.py
"""
import glob
import json
import os

from config import RESULTS_DIR

COLUMNS = ["run", "model", "T", "hops", "maxnb", "style", "used", "citers%", "halluc%",
           "flip_cited", "flip_random", "flip_control", "sufficiency", "flip_total"]


def load_runs(results_dir=RESULTS_DIR):
    """Description: Read every results/*.json (not summary files). Inputs: results_dir.
    Outputs: list of (name, dict). Dependencies: glob, json. Utilities: main()."""
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            runs.append((os.path.basename(path)[:-5], json.load(f)))
    return runs


def pct(value):
    """Description: Format a share as a percent; None/nan -> '-'. Inputs: value. Outputs: str.
    Dependencies: none. Utilities: make_row()."""
    return "-" if value is None or value != value else f"{100 * value:.1f}"


def make_row(name, run):
    """
    Description: One table row. Flip rates of cited/random/control/sufficiency are taken
        from the 'citers' group (nodes that cited at least one neighbor) - the fair
        comparison for hypothesis H1. flip_total is from 'all'.
    Inputs: name, run (dict from JSON). Outputs: list of str (same order as COLUMNS).
    Dependencies: pct(). Utilities: main().
    """
    s, st, m = run["settings"], run["stats"], run["metrics"]
    cit = m["citers"]
    return [name, s["model"], str(s["temperature"]), str(s["hops"]), str(s["max_neighbors"]),
            s["prompt_style"], str(st["used"]), pct(st["citer_share"]),
            pct(st["mean_hallucination_rate"]), pct(cit["cited_removal"]["flip_rate"]),
            pct(cit["random_removal"]["flip_rate"]), pct(cit["no_change"]["flip_rate"]),
            pct(cit["uncited_removal"]["sufficiency"]), pct(m["all"]["total_graph_removal"]["flip_rate"])]


def to_markdown(rows):
    """Description: Rows -> markdown table text. Inputs: rows (list of list of str).
    Reads COLUMNS. Outputs: str. Dependencies: none. Utilities: main()."""
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    """Description: Print and save the table. Inputs: none. Outputs: prints, writes results/summary.md."""
    rows = [make_row(n, r) for n, r in load_runs()]
    if not rows:
        print("No results/*.json yet. Run python run_sweep.py first.")
        return
    text = to_markdown(rows)
    print(text)
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
