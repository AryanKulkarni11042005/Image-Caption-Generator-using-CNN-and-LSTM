"""Tokenizer and Vocab class used throughout the caption-gen project.

This is the exact logic debugged in notebooks/01_data.ipynb, pulled into a
shared module so training notebooks and the deployment server import the
identical class instead of redefining it separately (and risking drift).
"""

import re
from collections import Counter

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z]+", s.lower())


class Vocab:
    def __init__(self, token_lists=None, min_freq: int = 5):
        # token_lists is optional so a Vocab can be constructed empty and
        # then populated via unpickling (pickle.load bypasses __init__ args
        # but still needs the class importable under this exact name/path).
        if token_lists is None:
            self.itos = [PAD, SOS, EOS, UNK]
            self.stoi = {w: i for i, w in enumerate(self.itos)}
            return

        counts = Counter(t for toks in token_lists for t in toks)
        self.itos = [PAD, SOS, EOS, UNK] + sorted(w for w, c in counts.items() if c >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, toks: list[str]) -> list[int]:
        return [self.stoi[SOS]] + [self.stoi.get(t, self.stoi[UNK]) for t in toks] + [self.stoi[EOS]]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids if i not in (0, 1, 2))
