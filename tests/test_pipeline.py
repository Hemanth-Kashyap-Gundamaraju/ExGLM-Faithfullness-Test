"""
Run with:  python -m unittest discover -s tests -v
(from the project folder). No internet and no GPU needed.
"""
import math
import unittest

import main
from citations import find_citations, hallucination_rate, is_explanation_consistent
from data_loader import cap_neighbors, clean_titles, get_k_hop_neighborhood, one_title
from interventions import apply_intervention
from metrics import MetricsTable, summarize_run
from prompts import build_prompt, prompt_title
from response_parser import class_probability, class_start_entry, find_label, parse_reply, split_response
from tests.fake_model import FakeClient, FakeData, make_graph


class TitleTests(unittest.TestCase):
    def test_prefix_removed_and_empty_title_falls_back(self):
        data = FakeData()
        self.assertEqual(one_title(data, 0), "Paper about case based number 0")
        self.assertTrue(one_title(data, 3).startswith("This is the abstract"))

    def test_quotes_inside_title_are_replaced(self):
        data = FakeData()
        data.title[1] = 'Title: A "quoted" word'
        self.assertEqual(one_title(data, 1), "A 'quoted' word")


class GraphTests(unittest.TestCase):
    def test_k_hop_and_cap(self):
        g = make_graph()
        self.assertEqual(set(get_k_hop_neighborhood(g, 0, 1)), {1, 2, 10, 11})
        full = get_k_hop_neighborhood(g, 0, 2)
        shown = cap_neighbors(g, 0, full, max_n=5)
        self.assertEqual(len(shown), 5)
        self.assertTrue(set(shown[:4]) == {1, 2, 10, 11})      # 1-hop first
        self.assertEqual(shown, cap_neighbors(g, 0, full, max_n=5))   # repeatable


class ParserTests(unittest.TestCase):
    def test_find_label(self):
        self.assertEqual(find_label("**Neural-Networks**"), "neural networks")
        self.assertEqual(find_label("Label: case_based"), "case based")
        self.assertIsNone(find_label("neural networks or theory"))
        self.assertIsNone(find_label("I do not know"))

    def test_split_response_skips_blank_lines(self):
        label, expl = split_response("\n\ntheory\nBecause.")
        self.assertEqual((label, expl), ("theory", "Because."))

    def test_probability(self):
        entry = {"token": "Neural", "logprob": math.log(0.6),
                 "top_logprobs": [{"token": "Neural", "logprob": math.log(0.6)},
                                  {"token": " neural", "logprob": math.log(0.2)},
                                  {"token": "Theory", "logprob": math.log(0.1)}]}
        self.assertAlmostEqual(class_probability(entry, "neural networks"), 0.8, places=5)
        self.assertIsNone(class_probability(None, "theory"))
        self.assertIsNone(class_probability({"top_logprobs": []}, "theory"))     # no information
        self.assertEqual(class_probability(entry, "case based"), 0.0)           # candidates given, none matches


class ClassStartTests(unittest.TestCase):
    def tok(self, token, p=0.9):
        return {"token": token, "logprob": math.log(p),
                "top_logprobs": [{"token": token, "logprob": math.log(p)}]}

    def test_finds_class_after_markdown_prefix(self):
        logprobs = [self.tok("**"), self.tok("Category"), self.tok(":"), self.tok(" Reinforcement", 0.7),
                    self.tok(" Learning"), self.tok("**"), self.tok("\n\n"), self.tok("Theory")]
        entry = class_start_entry(logprobs)
        self.assertEqual(entry["token"], " Reinforcement")
        reply = {"message": {"content": "**Category: Reinforcement Learning**\n\nTheory"}, "logprobs": logprobs}
        parsed = parse_reply(reply)
        self.assertEqual(parsed.label, "reinforcement learning")
        self.assertAlmostEqual(parsed.prob, 0.7, places=5)

    def test_only_first_line_is_searched(self):
        logprobs = [self.tok("Hmm"), self.tok("\n"), self.tok("Theory")]
        self.assertIsNone(class_start_entry(logprobs))
        self.assertIsNone(class_start_entry(None))

    def test_leading_blank_tokens_are_skipped(self):
        self.assertEqual(class_start_entry([self.tok("\n"), self.tok("Theory")])["token"], "Theory")


class CitationTests(unittest.TestCase):
    TITLES = {1: "Reinforcement Learning: A Survey", 2: "Surgery", 3: "Genetic Algorithms for Optimization"}

    def test_quoted_and_unquoted_and_hallucinated(self):
        title_of = self.TITLES.get
        expl = ('The article "Reinforcement Learning: A Survey" matters. Also genetic algorithms '
                'for optimization is mentioned, and "A Totally Made Up Paper Title" too.')
        res = find_citations(expl, [1, 2, 3], "Target Paper Title Here", title_of)
        self.assertEqual(res.valid_cited, [1, 3])
        self.assertEqual(len(res.bad_spans), 1)
        self.assertEqual(res.n_unmatchable, 1)           # "Surgery" is too short
        self.assertAlmostEqual(hallucination_rate(res), 1 / 3)

    def test_class_names_and_target_are_not_hallucinations(self):
        res = find_citations('It is "theory" and "Target Paper Title Here".', [1], "Target Paper Title Here", self.TITLES.get)
        self.assertEqual(res.bad_spans, [])

    def test_consistency(self):
        self.assertFalse(is_explanation_consistent("It is about probabilistic methods.", "neural networks"))
        self.assertTrue(is_explanation_consistent("It is about neural networks.", "neural networks"))


class InterventionTests(unittest.TestCase):
    def test_all_strategies(self):
        nb, cited = [1, 2, 3, 4, 5], [2, 4]
        self.assertEqual(apply_intervention(nb, cited, "cited_removal", 0), [1, 3, 5])
        self.assertEqual(apply_intervention(nb, cited, "uncited_removal", 0), [2, 4])
        self.assertEqual(apply_intervention(nb, cited, "total_graph_removal", 0), [])
        self.assertEqual(apply_intervention(nb, cited, "no_change", 0), nb)
        rnd = apply_intervention(nb, cited, "random_removal", 0)
        self.assertEqual(len(rnd), 3)
        self.assertTrue({2, 4} <= set(rnd))               # cited stay, uncited removed
        self.assertEqual(rnd, apply_intervention(nb, cited, "random_removal", 0))
        with self.assertRaises(ValueError):
            apply_intervention(nb, cited, "nonsense", 0)


class PromptTests(unittest.TestCase):
    def test_prompt_contains_titles_not_ids(self):
        titles = clean_titles(FakeData())
        prompt = build_prompt(0, titles, [1, 2])
        self.assertIn('"Paper about case based number 0" is connected with', prompt)
        self.assertIn("within", prompt)
        self.assertNotIn("Node 1", prompt)
        self.assertIn("has no known connections", build_prompt(0, titles, []))


class EndToEndTests(unittest.TestCase):
    def make_experiment(self, with_logprobs=True):
        data = FakeData()
        return main.Experiment(data=data, titles=clean_titles(data), graph=make_graph(),
                               client=FakeClient(with_logprobs))

    def test_full_run_with_fake_model(self):
        exp = self.make_experiment()
        nodes = [7, 8, 9, 10, 11]   # node 6 is skipped: its own class is theory, so a flip to theory is invisible
        results = main.run_all_nodes(exp, nodes)
        good = [r for r in results if r is not None]
        self.assertEqual(len(good), len(nodes))
        table = MetricsTable()
        for r in good:
            table.add(r)
        stats = summarize_run(good, 0)
        self.assertEqual(stats["citers"], len(good))           # fake model always cites its key neighbor
        cited = table.bucket("citers", "cited_removal")
        self.assertEqual(cited.n, len(good))
        self.assertEqual(cited.flip_rate(), 1.0)               # removing the key neighbor flips every answer
        self.assertEqual(table.bucket("citers", "random_removal").flip_rate(), 0.0)   # random removal keeps it
        self.assertEqual(table.bucket("citers", "uncited_removal").flip_rate(), 0.0)  # keeping only cited is enough
        self.assertEqual(table.bucket("citers", "total_graph_removal").flip_rate(), 1.0)
        self.assertEqual(table.bucket("citers", "no_change").flips, 0)
        self.assertFalse(math.isnan(cited.mean_drop()))
        report = main.format_report(table, stats)
        self.assertIn("CITERS", report)

    def test_works_without_probabilities(self):
        exp = self.make_experiment(with_logprobs=False)
        result = main.evaluate_node(6, exp)
        self.assertIsNotNone(result)
        self.assertIsNone(result.base_prob)
        table = MetricsTable()
        table.add(result)
        self.assertTrue(math.isnan(table.bucket("all", "cited_removal").mean_drop()))


if __name__ == "__main__":
    unittest.main()
