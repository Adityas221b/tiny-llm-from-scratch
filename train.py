"""Train the tiny GPT on the prepared token files.

Example:
    python train.py                      # final config (GPU, ~5 min on an RTX 5050 laptop)
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from tinyllm.model import GPT, GPTConfig


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seed", type=int, default=1337)
    # model
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "manual"],
                    help="sdpa = fused attention kernel, manual = explicit masked softmax")
    # optimisation
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # eval / logging
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=50)
    return ap.parse_args()


def lr_at(it: int, a) -> float:
    """Linear warmup, then cosine decay from lr to min_lr."""
    if it < a.warmup_iters:
        return a.lr * (it + 1) / a.warmup_iters
    progress = min(1.0, (it - a.warmup_iters) / max(1, a.max_iters - a.warmup_iters))
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (a.lr - a.min_lr)


def main() -> None:
    a = get_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    data = {s: np.fromfile(os.path.join(a.data_dir, f"{s}.bin"), dtype=np.uint16) for s in ("train", "val")}
    with open(os.path.join(a.data_dir, "tokenizer.json"), encoding="utf-8") as f:
        vocab_size = 256 + len(json.load(f)["merges"])

    def get_batch(split: str):
        d = data[split]
        ix = np.random.randint(0, len(d) - a.block_size - 1, size=a.batch_size)
        x = torch.from_numpy(np.stack([d[i:i + a.block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([d[i + 1:i + 1 + a.block_size] for i in ix]).astype(np.int64))
        return x.to(device), y.to(device)

    cfg = GPTConfig(vocab_size=vocab_size, block_size=a.block_size, n_layer=a.n_layer,
                    n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout, use_sdpa=a.attn == "sdpa")
    model = GPT(cfg).to(device)
    opt = model.configure_optimizer(a.lr, a.weight_decay)
    print(f"device={device} | params={model.num_params(non_embedding=False):,} | "
          f"train tokens={len(data['train']):,} val tokens={len(data['val']):,}")

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(a.eval_iters)
            for k in range(a.eval_iters):
                x, y = get_batch(split)
                losses[k] = model(x, y)[1].item()
            out[split] = losses.mean().item()
        model.train()
        return out

    history = {"config": vars(a), "model": asdict(cfg), "device": device, "train_curve": [], "evals": []}
    best_val = float("inf")
    t0 = time.time()
    for it in range(a.max_iters + 1):
        lr = lr_at(it, a)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % a.eval_interval == 0 or it == a.max_iters:
            l = estimate_loss()
            history["evals"].append({"iter": it, "train": l["train"], "val": l["val"], "time_s": time.time() - t0})
            print(f"[eval] iter {it:5d} | train {l['train']:.4f} | val {l['val']:.4f} | "
                  f"val ppl {math.exp(l['val']):.2f} | {time.time() - t0:.0f}s")
            if l["val"] < best_val:
                best_val = l["val"]
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "iter": it, "val_loss": best_val},
                           os.path.join(a.out_dir, "ckpt.pt"))
            if it == a.max_iters:
                break

        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        if it % a.log_interval == 0:
            history["train_curve"].append({"iter": it, "loss": loss.item(), "lr": lr})

    history["total_time_s"] = time.time() - t0
    history["best_val_loss"] = best_val
    with open(os.path.join(a.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    print(f"done in {history['total_time_s'] / 60:.1f} min | best val loss {best_val:.4f} "
          f"(ppl {math.exp(best_val):.2f})")


if __name__ == "__main__":
    main()
