# Do LLM explanations really depend on the neighbors they cite?
A faithfulness test for graph-language models (DS5622 Explainable AI, Track 2).

**Question.** When an explanation names a neighbor paper as a reason for a node's label,
does removing that neighbor change the model's prediction?
We compare removing the **cited** neighbors with removing the **same number of random
uncited** neighbors (main baseline), keeping **only** the cited ones (sufficiency),
removing **all** neighbors (text-only baseline), and a **no-change control** (how much
changes from sampling noise alone).

Target paper: Zhiqiang et al., *Judge and Improve: Towards a Better Reasoning of Knowledge
Graphs with Large Language Models* (EMNLP 2025, ExGLM). Its authors leave faithfulness for
future work ("Discussion on Plausibility versus Faithfulness").

## Folder map (one job per file)
| File | Job |
|---|---|
| `config.py` | all settings (ALL_CAPS globals, can be changed with environment variables) |
| `data_loader.py` | load Cora, clean titles, find / cap neighbors, pick test nodes |
| `prompts.py` | build the prompt (copy of the paper's Table 11 wording) |
| `ollama_client.py` | the only network code (retry, API key) |
| `response_cache.py` | save replies on disk so runs can stop and restart |
| `response_parser.py` | label, explanation and class probability from a reply |
| `citations.py` | which neighbors does an explanation cite (by title)? hallucinated titles? |
| `interventions.py` | the removal strategies |
| `metrics.py` | flip rate, probability drop, sufficiency, report text |
| `main.py` | run one experiment |
| `run_sweep.py` / `summarize_results.py` | sensitivity study / one results table |
| `download_cora.py`, `check_data.py`, `list_cloud_models.py`, `smoke_test.py` | setup and checks |
| `tests/` | unit tests that need no internet (`python -m unittest discover -s tests -v`) |

## How to run
1. `pip install -r requirements.txt`
2. `python download_cora.py` (3 small files from Hugging Face into `cora_data/cora/`)
3. `python check_data.py` (counts, labels, titles, neighbor sizes; read the output)
4. Create an Ollama API key at https://ollama.com/settings/keys and set it as an
   environment variable (never write it in a file or commit it):
   - Linux/Mac: `export OLLAMA_API_KEY=...`   - Windows PowerShell: `$env:OLLAMA_API_KEY="..."`
   - Colab: `import os; from google.colab import userdata; os.environ["OLLAMA_API_KEY"] = userdata.get("OLLAMA_API_KEY")`
5. `python list_cloud_models.py`, then set `OLLAMA_MODEL` (env var or `config.py`) to a Qwen
   name exactly as printed.
6. `python smoke_test.py` (one node; shows prompt, answer, parsed label, citations)
7. `NUM_TEST_NODES=5 python main.py` (small run), then `python main.py` (all 542 test nodes)
8. `python run_sweep.py` then `python summarize_results.py` (sensitivity table)

### Which model?
Two routes, same code:
- **Proposal models (Qwen2.5-1.5B and 7B, 4-bit) on Google Colab with a local Ollama** (free T4 GPU):
  `OLLAMA_URL=http://localhost:11434/api/chat OLLAMA_MODEL=qwen2.5:1.5b`, and
  `OTHER_MODEL=qwen2.5:7b` for the size run. No API key needed.
- **Ollama cloud API.** Our key's cloud list (checked with `list_cloud_models.py`) had no Qwen model,
  so use another open model pair, for example `gpt-oss:20b` and `gpt-oss:120b` (reasoning models:
  set `THINK_MODE=low`; probabilities are then not reliable and are reported as `nan`).

## Metrics
- **Prediction flip rate**: share of nodes whose predicted label changes after the intervention.
- **Probability drop**: probability of the *original* label before minus after (from first-token
  probabilities, only if the server returns them; otherwise reported as `nan`).
- **Sufficiency**: share of predictions kept when only the cited neighbors stay (`1 - flip rate`).
- **Citation hallucination rate**: share of quoted titles in the explanation that match no neighbor
  shown in the prompt (an upper bound: a quoted phrase that is not a title also counts).
- All metrics are shown for `all`, `correct`, `incorrect` predictions and for `citers`
  (nodes that cited at least one shown neighbor). **H1 must be judged on `citers`**: for nodes
  that cite nobody, cited removal changes nothing by construction.

## Settings (sensitivity study)
`TEMPERATURE` (0.0 / 0.7), `HOPS` (1 / 2), `MAX_NEIGHBORS` (5 / 10 / 20), `PROMPT_STYLE`
(A = paper wording / B = also asks to name articles by title), `OLLAMA_MODEL`, `RUN_TAG`.
One setting is changed at a time from the base setting.

## Deviations from the proposal (write these in the report)
- ExGLM is **not** re-trained: it trains LLaMA with GNN embeddings and Judge-and-Improve.
  We only prompt an open model with the paper's prompt format (titles of 1/2-hop neighbors).
- Model: see "Which model?". If the cloud route is used, the models differ from the proposal's
  Qwen2.5-1.5B/7B. LoRA fine-tuning (optional in the proposal) is not done.
- Data: Cora with text from `Graph-COM/Text-Attributed-Graphs` (2,708 nodes, 5,278 undirected
  edges, dataset's own 542 test nodes). The paper reports 5,429 edges and uses the split of Ye et al.
- Added a no-change control and the `citers` group (not in the proposal).

## Known limitations
- Cora has noisy or repeated titles; 42 empty titles are replaced by the start of the abstract;
  titles shorter than 12 characters cannot be matched in citations.
- Cloud models may be updated by the provider; the cache and `RUN_TAG` make our own runs repeatable,
  but not future re-runs.
- At T=0.7 the `no_change` control uses a different seed (full resampling noise). `RUN_TAG=run2` only
  changes the cache key, not the seed: it repeats the same setting, so the sweep shows how repeatable
  a run is (the two T=0.7 runs differ even with the same seed).

## LLM-use disclosure (edit before submitting)
Code was written with help from Claude (Anthropic) and Gemini: project structure, debugging,
reading the ExGLM paper. We verified it by (1) unit tests with a fake model (`tests/`),
(2) `check_data.py` numbers, (3) reading the prompt and answer of single nodes with `smoke_test.py`,
(4) checking by hand that removed neighbors really disappear from the prompt.
Both team members can explain every file.

## Speed note
The first (base) answer is generated with `NUM_PREDICT=600` tokens so the full explanation can be read for citations. The five re-runs after each removal only need the label line, so they use `NUM_PREDICT_RERUN=40` tokens.
