"""Measure generation throughput with and without the KV cache.

Without a cache, producing token t re-runs attention and MLPs over the whole context
(O(T) work per token, O(T^2) attention). With the cache, each step processes one position and
attends over stored keys/values.

    python bench_generate.py --ckpt out/ckpt.pt --device cuda
"""
import argparse
import json
import os
import time

import torch

from tinyllm.model import GPT, GPTConfig


def bench(model, device, prompt_len, new_tokens, use_cache, repeats):
    idx = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
    times = []
    for r in range(repeats + 1):          # first run is a warm-up
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(idx, new_tokens, temperature=1.0, top_k=40, use_cache=use_cache)
        if device == "cuda":
            torch.cuda.synchronize()
        if r:
            times.append(time.perf_counter() - t0)
    return new_tokens / (sum(times) / len(times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    ap.add_argument("--prompt_len", type=int, default=8)
    ap.add_argument("--new_tokens", type=int, default=240, help="stays inside the context window")
    ap.add_argument("--long_tokens", type=int, default=1000, help="runs past the window (cache rebuilds)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="runs/kv_cache_bench.json")
    a = ap.parse_args()

    res = {"ckpt": a.ckpt, "prompt_len": a.prompt_len, "results": []}
    for device in a.devices:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        ckpt = torch.load(a.ckpt, map_location=device)
        model = GPT(GPTConfig(**{**ckpt["config"], "dropout": 0.0})).to(device).eval()
        model.load_state_dict(ckpt["model"])
        for n in (a.new_tokens, a.long_tokens):
            no = bench(model, device, a.prompt_len, n, False, a.repeats)
            yes = bench(model, device, a.prompt_len, n, True, a.repeats)
            row = {"device": device, "new_tokens": n, "tok_per_s_no_cache": no, "tok_per_s_kv_cache": yes,
                   "speedup": yes / no}
            res["results"].append(row)
            print(f"{device:4s} | {n:5d} tokens | no cache {no:8.1f} tok/s | KV cache {yes:8.1f} tok/s | "
                  f"{yes / no:.2f}x")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
