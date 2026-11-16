"""
smoke_test.py - ONE job: try ONE real node end to end and show everything, so you can
see that the key, model, parsing and probabilities work BEFORE a long run.
    python smoke_test.py
"""
from citations import find_citations
from config import OLLAMA_MODEL
from data_loader import cap_neighbors, get_k_hop_neighborhood, get_test_nodes, load_cora_with_text
from ollama_client import OllamaClient
from prompts import build_prompt, prompt_title
from response_parser import class_start_entry, parse_reply


def main():
    """Description: One node, one call, print prompt / reply / parsed result.
    Inputs: none. Outputs: prints. Dependencies: data_loader, prompts, ollama_client, parser, citations."""
    data, titles, graph = load_cora_with_text()
    node = get_test_nodes(data, 1)[0]
    shown = cap_neighbors(graph, node, get_k_hop_neighborhood(graph, node))
    prompt = build_prompt(node, titles, shown)
    print("MODEL:", OLLAMA_MODEL, "| node:", node, "| neighbors shown:", len(shown))
    print("\n--- PROMPT ---\n" + prompt)
    reply = OllamaClient(use_cache=False).chat(prompt)
    print("\n--- RAW ANSWER ---\n" + reply["message"]["content"])
    parsed = parse_reply(reply)
    print("\n--- PARSED ---")
    print("label:", parsed.label, "| probability of label:", parsed.prob)
    print("logprobs returned by server:", bool(reply.get("logprobs")),
          "| class-name token found:", class_start_entry(reply.get("logprobs")) is not None)
    title_of = lambda n: prompt_title(titles, n)
    found = find_citations(parsed.explanation, shown, title_of(node), title_of)
    print("cited neighbors:", found.valid_cited, "| quoted titles matching nothing:", len(found.bad_spans))
    print("true label:", data.label_texts[int(data.y[node])])


if __name__ == "__main__":
    main()
