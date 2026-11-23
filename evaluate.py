"""Deterministic loss / perplexity over an entire split (non-overlapping block_size windows).

Also reports bits-per-character, which (unlike per-token perplexity) is comparable
across tokenizers, e.g. against character-level models on the same corpus.
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from tinyllm.model import GPT, GPTConfig
from tinyllm.tokenizer import BPETokenizer


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default=None, help="optional JSON file to write results to")
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    ckpt = torch.load(a.ckpt, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])

    ids = np.fromfile(os.path.join(a.data_dir, f"{a.split}.bin"), dtype=np.uint16).astype(np.int64)
    T = cfg.block_size
    n_win = (len(ids) - 1) // T
    x = torch.from_numpy(ids[: n_win * T]).view(n_win, T)
    y = torch.from_numpy(ids[1 : n_win * T + 1]).view(n_win, T)

    total_nll, total_tok = 0.0, 0
    for i in range(0, n_win, a.batch_size):
        xb, yb = x[i:i + a.batch_size].to(device), y[i:i + a.batch_size].to(device)
        logits, _ = model(xb)
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1),
                                                reduction="sum")
        total_nll += nll.item()
        total_tok += yb.numel()

    tok = BPETokenizer.load(os.path.join(a.data_dir, "tokenizer.json"))
    n_chars = len(tok.decode(y.reshape(-1).tolist()))
    loss = total_nll / total_tok
    res = {"split": a.split, "checkpoint_iter": ckpt["iter"], "tokens": total_tok, "chars": n_chars,
           "loss_nats_per_token": loss, "perplexity": math.exp(loss),
           "bits_per_char": total_nll / math.log(2) / n_chars}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
