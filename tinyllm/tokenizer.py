"""Byte-level BPE tokenizer, written from scratch.

Training follows the GPT-2 recipe in miniature:
  1. Pre-split text into chunks with a regex (words, numbers, punctuation, whitespace)
     so merges never cross word boundaries.
  2. Represent every chunk as a sequence of raw UTF-8 bytes (ids 0..255), so any
     string is encodable and decoding is lossless.
  3. Repeatedly merge the most frequent adjacent pair into a new token id.

Pair counts are computed over *unique* chunks weighted by frequency, which keeps
training on a ~1 MB corpus to a few seconds in pure Python.
"""
from __future__ import annotations

import json
import re
from collections import Counter

# Simplified GPT-2 style pre-tokenization pattern (no \p{L} support in stdlib `re`).
SPLIT_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256
        chunk_counts = Counter(SPLIT_PATTERN.findall(text))
        words = [(list(c.encode("utf-8")), n) for c, n in chunk_counts.items()]
        self.merges, self.vocab, self._cache = {}, {i: bytes([i]) for i in range(256)}, {}
        for new_id in range(256, vocab_size):
            stats: Counter = Counter()
            for ids, n in words:
                for a, b in zip(ids, ids[1:]):
                    stats[(a, b)] += n
            if not stats:
                break
            pair = max(stats, key=stats.get)
            words = [(_merge(ids, pair, new_id), n) for ids, n in words]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (new_id - 256) % 64 == 0:
                print(f"merge {new_id - 255}: {pair} -> {self.vocab[new_id]!r} (count {stats[pair]})")

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # apply the earliest-learned merge present in the chunk
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": [[a, b, i] for (a, b), i in self.merges.items()]}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for a, b, i in data["merges"]:  # stored in learn order
            tok.merges[(a, b)] = i
            tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
        return tok
