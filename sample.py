"""Generate text from a trained checkpoint.

Example:
    python sample.py --prompt "ROMEO:" --temperature 0.8 --top_k 40 --max_new_tokens 200
"""
import argparse
import os

import torch

from tinyllm.model import GPT, GPTConfig
from tinyllm.tokenizer import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    torch.manual_seed(a.seed)
    ckpt = torch.load(a.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(a.tokenizer)

    idx = torch.tensor([tok.encode(a.prompt)], dtype=torch.long, device=device)
    for i in range(a.num_samples):
        out = model.generate(idx, a.max_new_tokens, temperature=a.temperature, top_k=a.top_k)
        print(tok.decode(out[0].tolist()))
        if i < a.num_samples - 1:
            print("-" * 60)


if __name__ == "__main__":
    main()
