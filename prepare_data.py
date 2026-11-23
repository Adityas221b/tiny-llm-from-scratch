"""Download TinyShakespeare, train the BPE tokenizer on the train split, and encode both splits.

Outputs (in data/):
  input.txt        raw corpus (~1.1 MB, public domain)
  tokenizer.json   learned BPE merges
  train.bin/val.bin  uint16 token ids
"""
import argparse
import os
import time
import urllib.request

import numpy as np

from tinyllm.tokenizer import BPETokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    path = os.path.join(args.data_dir, "input.txt")
    if not os.path.exists(path):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    print(f"corpus: {len(text):,} characters")

    # Contiguous split: first 90% train, last 10% val. The tokenizer only sees train text.
    n = int(len(text) * (1 - args.val_frac))
    train_text, val_text = text[:n], text[n:]

    t0 = time.time()
    tok = BPETokenizer()
    tok.train(train_text, args.vocab_size, verbose=True)
    print(f"trained BPE (vocab {tok.vocab_size}) in {time.time() - t0:.1f}s")
    tok.save(os.path.join(args.data_dir, "tokenizer.json"))

    for name, split in [("train", train_text), ("val", val_text)]:
        ids = np.array(tok.encode(split), dtype=np.uint16)
        ids.tofile(os.path.join(args.data_dir, f"{name}.bin"))
        print(f"{name}: {len(split):,} chars -> {len(ids):,} tokens ({len(split) / len(ids):.2f} chars/token)")


if __name__ == "__main__":
    main()
