"""
metrics.py - ONE job: collect per-node results and compute the metrics
(flip rate, probability drop, sufficiency, hallucination rate, ...).
Pure logic: no network, no files.
"""
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import GROUPS, STRATEGIES

Outcome = Optional[Tuple[bool, Optional[float]]]   # (label flipped?, probability drop or None)


@dataclass
class NodeResult:
    """Description: Everything we learned about one test node.
    outcomes[strategy] is None when the model gave an unreadable answer for that prompt."""
    node_id: int
    base_label: str
    is_correct: bool
    base_prob: Optional[float]
    valid_cited: List[int]
    n_bad_spans: int
    hallucination_rate: float
    is_consistent: bool
    n_full_neighborhood: int
    n_shown: int
    outcomes: Dict[str, Outcome] = field(default_factory=dict)


@dataclass
class Bucket:
    """Description: Running totals for one (group, strategy) cell."""
    n: int = 0
    flips: int = 0
    drop_sum: float = 0.0
    drop_n: int = 0

    def add(self, flipped, drop):
        """Description: Add one node's outcome. Inputs: flipped (bool), drop (float or None).
        Outputs: none (changes this bucket). Dependencies: none. Utilities: MetricsTable.add()."""
        self.n += 1
        self.flips += int(flipped)
        if drop is not None:
            self.drop_sum += drop
            self.drop_n += 1

    def flip_rate(self):
        """Description: share of nodes whose label changed. Outputs: float or nan."""
        return self.flips / self.n if self.n else math.nan

    def mean_drop(self):
        """Description: average drop of the ORIGINAL class probability. Outputs: float or nan."""
        return self.drop_sum / self.drop_n if self.drop_n else math.nan


def groups_of(result):
    """Description: Which groups a node belongs to (all, correct/incorrect, citers).
    Inputs: NodeResult. Reads global GROUPS. Outputs: list of str.
    Dependencies: none. Utilities: MetricsTable.add()."""
    groups = ["all", "correct" if result.is_correct else "incorrect"]
    if result.valid_cited:
        groups.append("citers")
    return [g for g in groups if g in GROUPS]


class MetricsTable:
    """
    Description: Buckets for every (group, strategy). Feed it NodeResults, then read it.
    Inputs: none. Reads globals GROUPS, STRATEGIES.
    Outputs: bucket(), to_dict().
    Dependencies: Bucket, groups_of().
    Utilities: main.py, report.py.
    """

    def __init__(self):
        self.cells = {g: {s: Bucket() for s in STRATEGIES} for g in GROUPS}

    def add(self, result):
        """Description: Add one NodeResult to all its groups. Outputs: none."""
        for strategy, outcome in result.outcomes.items():
            if outcome is None:
                continue
            flipped, drop = outcome
            for group in groups_of(result):
                self.cells[group][strategy].add(flipped, drop)

    def bucket(self, group, strategy):
        """Description: One cell. Inputs: group, strategy. Outputs: Bucket."""
        return self.cells[group][strategy]

    def to_dict(self):
        """Description: Plain numbers for saving as JSON. Outputs: dict."""
        out = {}
        for g in GROUPS:
            out[g] = {s: {"n": b.n, "flip_rate": b.flip_rate(), "mean_prob_drop": b.mean_drop(),
                          "sufficiency": (1 - b.flip_rate()) if s == "uncited_removal" and b.n else None}
                      for s, b in self.cells[g].items()}
        return out


def summarize_run(results, n_skipped):
    """
    Description: Run-level numbers.
    Inputs: results (list of NodeResult), n_skipped (nodes with unreadable base answer or error).
    Outputs: dict: used, skipped, accuracy, citers, citer_share, hallucination_rate (mean),
        inconsistent, capped_nodes, mean_cited.
    Dependencies: statistics. Utilities: main.py.
    """
    used = len(results)
    citers = [r for r in results if r.valid_cited]
    return {
        "used": used,
        "skipped": n_skipped,
        "accuracy": sum(r.is_correct for r in results) / used if used else math.nan,
        "citers": len(citers),
        "citer_share": len(citers) / used if used else math.nan,
        "mean_cited_per_citer": statistics.mean(len(r.valid_cited) for r in citers) if citers else math.nan,
        "mean_hallucination_rate": statistics.mean(r.hallucination_rate for r in results) if results else math.nan,
        "nodes_with_bad_quotes": sum(r.n_bad_spans > 0 for r in results),
        "inconsistent_explanations": sum(not r.is_consistent for r in results),
        "capped_nodes": sum(r.n_full_neighborhood > r.n_shown for r in results),
    }


def format_report(table, stats):
    """
    Description: Human-readable text report (this is what you paste into the report).
    Inputs: MetricsTable, stats (from summarize_run). Reads globals GROUPS, STRATEGIES.
    Outputs: str.
    Dependencies: none. Utilities: main.py.
    """
    lines = [f"Nodes used: {stats['used']}   skipped: {stats['skipped']}   "
             f"accuracy: {stats['accuracy']:.1%}",
             f"Nodes citing >=1 shown neighbor: {stats['citers']} ({stats['citer_share']:.1%}); "
             f"mean cited per citer: {stats['mean_cited_per_citer']:.2f}",
             f"Mean citation hallucination rate: {stats['mean_hallucination_rate']:.1%} "
             f"(nodes with a quoted title that matches no shown neighbor: {stats['nodes_with_bad_quotes']})",
             f"Explanation disagrees with own label: {stats['inconsistent_explanations']}",
             f"Nodes whose neighbor list was capped: {stats['capped_nodes']}"]
    for group in GROUPS:
        lines.append(f"\n=== {group.upper()} ===")
        for strategy in STRATEGIES:
            b = table.bucket(group, strategy)
            if b.n == 0:
                continue
            extra = f"  sufficiency={1 - b.flip_rate():.1%}" if strategy == "uncited_removal" else ""
            lines.append(f"{strategy:20s} n={b.n:4d} flip={b.flip_rate():6.1%} "
                         f"prob_drop={b.mean_drop():7.3f}{extra}")
    return "\n".join(lines)
