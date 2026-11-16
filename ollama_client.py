"""
ollama_client.py - ONE job: talk to Ollama (cloud or local) over HTTP.
All network code of the project is in this file.
"""
import time

import requests

from config import (MAX_RETRIES, NUM_PREDICT, OLLAMA_API_KEY, OLLAMA_MODEL,
                    OLLAMA_URL, REQUEST_TIMEOUT, RETRY_SLEEP_SECONDS, SEED,
                    TEMPERATURE, THINK_MODE, TOP_LOGPROBS, USE_LOGPROBS)
from response_cache import load_cached_reply, make_cache_key, save_cached_reply

RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
THINK_LEVELS = ("low", "medium", "high")


class FatalOllamaError(RuntimeError):
    """Description: Error that retrying cannot fix (wrong key, wrong model name).
    The experiment stops at once instead of skipping every node quietly."""


def build_headers(api_key=OLLAMA_API_KEY):
    """Description: HTTP headers (adds the key if there is one).
    Inputs: api_key. Outputs: dict. Dependencies: none. Utilities: OllamaClient."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def build_request_body(prompt, model=OLLAMA_MODEL, temperature=TEMPERATURE, seed=SEED,
                       num_predict=NUM_PREDICT, think_mode=THINK_MODE,
                       use_logprobs=USE_LOGPROBS):
    """
    Description: The JSON sent to Ollama's /api/chat.
    Inputs: prompt + settings (defaults from config globals).
    Outputs: dict.
    Dependencies: none (pure function, easy to test).
    Utilities: OllamaClient.chat().
    """
    body = {"model": model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict}}
    if use_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = TOP_LOGPROBS
    if think_mode == "off":
        body["think"] = False
    elif think_mode in THINK_LEVELS:          # reasoning models such as gpt-oss take a level
        body["think"] = think_mode
    return body


def should_retry(status_code):
    """Description: Is this HTTP status worth trying again (rate limit / server error)?
    Inputs: status_code. Outputs: bool. Dependencies: none. Utilities: OllamaClient._post()."""
    return status_code in RETRY_STATUS_CODES


class OllamaClient:
    """
    Description: Sends prompts to Ollama, retries on temporary errors, and caches replies.
    Inputs (constructor): url, api_key, model, use_cache. Reads config globals as defaults.
    Outputs: chat() returns the raw reply dict from Ollama.
    Dependencies: requests, response_cache, build_headers(), build_request_body().
    Utilities: used by main.py, smoke_test.py.
    """

    def __init__(self, url=OLLAMA_URL, api_key=OLLAMA_API_KEY, model=OLLAMA_MODEL, use_cache=True):
        if "ollama.com" in url and not api_key:
            raise FatalOllamaError("OLLAMA_API_KEY is not set. Create a key at "
                               "https://ollama.com/settings/keys and set it as an "
                               "environment variable (never write it in a file).")
        self.url, self.api_key, self.model, self.use_cache = url, api_key, model, use_cache

    def _post(self, body):
        """Description: One HTTP call with retries. Inputs: body (dict).
        Outputs: reply dict. Dependencies: requests, should_retry()."""
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(self.url, headers=build_headers(self.api_key),
                                         json=body, timeout=REQUEST_TIMEOUT)
                if response.status_code in (401, 403):
                    raise FatalOllamaError(f"Ollama refused the request ({response.status_code}). "
                                           "Check OLLAMA_API_KEY and the model name.")
                if should_retry(response.status_code):
                    last_error = RuntimeError(f"HTTP {response.status_code}")
                elif 400 <= response.status_code < 500:
                    raise FatalOllamaError(f"Ollama error {response.status_code}: {response.text[:300]}")
                else:
                    return response.json()
            except (requests.ConnectionError, requests.Timeout) as error:
                last_error = error
            time.sleep(RETRY_SLEEP_SECONDS * (attempt + 1))
        raise RuntimeError(f"Ollama call failed after {MAX_RETRIES + 1} tries: {last_error}")

    def chat(self, prompt, seed_offset=0, num_predict=None):
        """
        Description: Reply for one prompt (from cache if we have it).
        Inputs: prompt, seed_offset (0 normally; 1 for the no-change control, so the
            control is a truly new sample at temperature > 0), num_predict (max new
            tokens; default NUM_PREDICT). Reads globals SEED, NUM_PREDICT.
        Outputs: reply dict.
        Dependencies: cache functions, _post().
        """
        seed = SEED + seed_offset
        tokens = num_predict or NUM_PREDICT
        key = (make_cache_key(prompt, model=self.model, seed=seed, num_predict=tokens)
               if self.use_cache else None)
        if key:
            cached = load_cached_reply(key)
            if cached is not None:
                return cached
        reply = self._post(build_request_body(prompt, model=self.model, seed=seed, num_predict=tokens))
        if key:
            save_cached_reply(key, reply)
        return reply
