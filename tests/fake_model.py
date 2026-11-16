"""
fake_model.py - a pretend Ollama client + tiny pretend dataset for tests.
No internet, no torch. It lets us check that the whole pipeline runs.
"""
import networkx as nx

CLASSES = ["case based", "genetic algorithms", "neural networks", "probabilistic methods",
           "reinforcement learning", "rule learning", "theory"]


class FakeTensor(list):
    """A list that also has .tolist() and int(), like the bits of a tensor we use."""
    def tolist(self):
        return list(self)


class FakeMask:
    def __init__(self, flags):
        self.flags = flags

    def nonzero(self, as_tuple=False):
        class _Idx(list):
            def view(self, _):
                return self
        return _Idx([i for i, f in enumerate(self.flags) if f])


class FakeData:
    """Looks like the PyG Data object: title, abs, y, test_mask, num_nodes."""
    def __init__(self, n=12):
        self.num_nodes = n
        self.title = [f"Title: Paper about {CLASSES[i % 7]} number {i}" for i in range(n)]
        self.title[3] = ""                       # an empty title, on purpose
        self.abs = [f"Abstract: This is the abstract of paper {i}" for i in range(n)]
        self.y = [i % 7 for i in range(n)]
        self.test_mask = FakeMask([i >= n // 2 for i in range(n)])


def make_graph(n=12):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        graph.add_edge(i, (i + 1) % n)
        graph.add_edge(i, (i + 2) % n)
    return graph


class FakeClient:
    """
    Pretend model with ONE key neighbor per paper (paper N -> paper N+1).
    It answers the class named in the target title if the key neighbor is in the
    prompt, otherwise 'theory' (so removing the key neighbor flips the answer).
    Its explanation quotes the key neighbor, so the pipeline should find it.
    """
    def __init__(self, with_logprobs=True):
        self.calls = 0
        self.with_logprobs = with_logprobs

    def chat(self, prompt, seed_offset=0, num_predict=None):
        self.calls += 1
        link = prompt.split("\n")[1]
        target = link.split('" is connected with')[0].split('" has no known')[0].strip('"')
        own_class = next(c for c in CLASSES if c in target)
        number = int(target.rsplit("number ", 1)[1])
        key = (number + 1) % 12
        key_title = f"Paper about {CLASSES[key % 7]} number {key}"
        has_key = f'"{key_title}"' in link
        label = own_class if has_key else "theory"
        text = f'{label}\nThe article "{target}" is linked to "{key_title}", so it is {label}.' \
            if has_key else f"{label}\nNothing useful nearby, so {label}."
        reply = {"message": {"content": text}}
        if self.with_logprobs:
            first_word = label.split()[0].capitalize()
            reply["logprobs"] = [{"token": first_word, "logprob": -0.1,
                                  "top_logprobs": [{"token": first_word, "logprob": -0.1},
                                                   {"token": " other", "logprob": -3.0}]}]
        return reply
