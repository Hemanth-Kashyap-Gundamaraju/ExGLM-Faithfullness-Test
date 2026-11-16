"""Tests for the network code (with requests.post replaced by a fake) and the sweep helpers."""
import os
import tempfile
import unittest
from unittest import mock

import ollama_client
import response_cache
import run_sweep
import summarize_results


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code, self._payload, self.text = status, payload or {}, "body"

    def json(self):
        return self._payload


class ClientTests(unittest.TestCase):
    def test_request_body(self):
        body = ollama_client.build_request_body("hi", model="m", temperature=0.7, seed=5,
                                                num_predict=10, think_mode="off", use_logprobs=1)
        self.assertEqual(body["options"], {"temperature": 0.7, "seed": 5, "num_predict": 10})
        self.assertIs(body["think"], False)
        self.assertTrue(body["logprobs"])
        level = ollama_client.build_request_body("hi", think_mode="low", use_logprobs=0)
        self.assertEqual(level["think"], "low")
        plain = ollama_client.build_request_body("hi", think_mode="default", use_logprobs=0)
        self.assertNotIn("think", plain)
        self.assertNotIn("logprobs", plain)

    def test_headers(self):
        self.assertEqual(ollama_client.build_headers("abc")["Authorization"], "Bearer abc")
        self.assertNotIn("Authorization", ollama_client.build_headers(""))

    def test_missing_key_is_fatal_for_cloud_url(self):
        with self.assertRaises(ollama_client.FatalOllamaError):
            ollama_client.OllamaClient(url="https://ollama.com/api/chat", api_key="")

    def test_retry_then_success_and_cache(self):
        reply = {"message": {"content": "theory\nx"}}
        calls = [FakeResponse(429), FakeResponse(200, reply)]
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(ollama_client, "RETRY_SLEEP_SECONDS", 0), \
                mock.patch.object(response_cache, "CACHE_DIR", tmp), \
                mock.patch("ollama_client.requests.post", side_effect=calls) as post:
            client = ollama_client.OllamaClient(url="http://localhost:11434/api/chat", api_key="")
            self.assertEqual(client.chat("prompt"), reply)
            self.assertEqual(client.chat("prompt"), reply)       # second time: from cache
            self.assertEqual(post.call_count, 2)                 # 1 failed + 1 ok, none for the cached call

    def test_wrong_model_is_fatal(self):
        with mock.patch("ollama_client.requests.post", return_value=FakeResponse(404)):
            client = ollama_client.OllamaClient(url="http://x/api/chat", api_key="", use_cache=False)
            with self.assertRaises(ollama_client.FatalOllamaError):
                client.chat("p")

    def test_num_predict_is_sent_and_cached_separately(self):
        reply = {"message": {"content": "theory"}}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(response_cache, "CACHE_DIR", tmp), \
                mock.patch("ollama_client.requests.post", return_value=FakeResponse(200, reply)) as post:
            client = ollama_client.OllamaClient(url="http://localhost:11434/api/chat", api_key="")
            client.chat("p", num_predict=40)
            client.chat("p", num_predict=40)      # cached
            client.chat("p")                      # other length -> new call
            self.assertEqual(post.call_count, 2)
            self.assertEqual(post.call_args_list[0].kwargs["json"]["options"]["num_predict"], 40)

    def test_seed_offset_changes_cache_key(self):
        a = response_cache.make_cache_key("p", seed=42)
        b = response_cache.make_cache_key("p", seed=43)
        self.assertNotEqual(a, b)


class SweepTests(unittest.TestCase):
    def test_run_names(self):
        self.assertEqual(run_sweep.make_run_name({}), "base")
        self.assertEqual(run_sweep.make_run_name({"TEMPERATURE": "0.7"}), "temperature-0p7")
        self.assertEqual(run_sweep.make_run_name({"OLLAMA_MODEL": "qwen3-next:80b"}), "ollama_model-qwen3-next80b")

    def test_summary_row(self):
        cell = lambda flip, suff=None: {"n": 5, "flip_rate": flip, "mean_prob_drop": 0.1, "sufficiency": suff}
        run = {"settings": {"model": "m", "temperature": 0.0, "hops": 2, "max_neighbors": 10, "prompt_style": "A"},
               "stats": {"used": 10, "citer_share": 0.5, "mean_hallucination_rate": 0.1},
               "metrics": {"citers": {"cited_removal": cell(0.4), "random_removal": cell(0.1),
                                      "no_change": cell(0.0), "uncited_removal": cell(0.2, 0.8)},
                           "all": {"total_graph_removal": cell(0.3)}}}
        row = summarize_results.make_row("base", run)
        self.assertEqual(len(row), len(summarize_results.COLUMNS))
        self.assertIn("40.0", row)
        self.assertIn("80.0", row)
        self.assertIn("|", summarize_results.to_markdown([row]))


if __name__ == "__main__":
    unittest.main()
