"""Mechanistic interpretability: find induction heads, test them causally, and track when they form.

Induction heads (Elhage et al. 2021; Olsson et al. 2022) implement in-context copying:
"if A was followed by B earlier, predict B after the next A". They are detected with
sequences of random tokens repeated twice, [x_1..x_L, x_1..x_L]: the first half is
unpredictable, but a model with induction heads can predict the second half almost perfectly.

  * induction score of a head = mean attention from position i (second half) to position i-L+1,
    i.e. to the token that followed the previous occurrence of the current token
  * previous-token score = mean attention from i to i-1 (the other half of the circuit)
  * causal test: zero out the top-k induction heads and measure how much in-context copying
    degrades, vs. zeroing k random other heads as a control
  * dynamics (--snapshots DIR): repeat the measurements on checkpoints saved during training

    python interpret.py --ckpt out/ckpt.pt
    python interpret.py --ckpt out/ckpt.pt --snapshots out/snapshots --history out/history.json
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from tinyllm.model import GPT, GPTConfig


def load(path: str, device: str) -> GPT:
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**{**ckpt["config"], "use_sdpa": False, "dropout": 0.0})
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


def token_pool(data_dir: str, min_count: int = 100) -> torch.Tensor:
    """Tokens that actually occur in the training data. Sampling uniformly from the whole vocab would
    include byte tokens never seen in training (e.g. control characters), which pushes the model far
    off-distribution (loss ~15 nats instead of ~ln(vocab))."""
    ids = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    counts = np.bincount(ids, minlength=1)
    return torch.from_numpy(np.nonzero(counts >= min_count)[0].astype(np.int64))


def repeated_batch(pool: torch.Tensor, L: int, B: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = pool[torch.randint(0, len(pool), (B, L), generator=g)]
    return torch.cat([x, x], dim=1).to(device)


def natural_batch(data_dir: str, L: int, B: int, seed: int, device: str) -> torch.Tensor:
    ids = torch.from_numpy(np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16).astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - L, (B,), generator=g)
    x = torch.stack([ids[s:s + L] for s in starts])
    return torch.cat([x, x], dim=1).to(device)


@torch.no_grad()
def measure(model: GPT, seq: torch.Tensor, L: int, head_mask: dict | None = None) -> dict:
    """Return per-half loss and per-head induction / previous-token scores."""
    attns = [b.attn for b in model.blocks]
    for i, a in enumerate(attns):
        a.store_attn = True
        a.head_mask = None if head_mask is None or i not in head_mask else head_mask[i]
    logits, _ = model(seq[:, :-1])
    nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), seq[:, 1:].reshape(-1),
                          reduction="none").view(seq.size(0), -1)
    # predictions of tokens 1..L-1 (first half) vs tokens L+1..2L-1 (second half, copyable)
    first, second = nll[:, : L - 1].mean().item(), nll[:, L:].mean().item()

    T = seq.size(1) - 1
    q = torch.arange(L, T)                                   # query positions in the second half
    ind = np.zeros((len(attns), model.cfg.n_head))
    prev = np.zeros_like(ind)
    for li, a in enumerate(attns):
        A = a.last_attn.float()                              # (B, h, T, T)
        ind[li] = A[:, :, q, q - L + 1].mean(dim=(0, 2)).cpu().numpy()
        p = torch.arange(1, T)
        prev[li] = A[:, :, p, p - 1].mean(dim=(0, 2)).cpu().numpy()
        a.store_attn, a.last_attn, a.head_mask = False, None, None
    return {"loss_first_half": first, "loss_second_half": second, "induction": ind, "prev_token": prev}


def mask_for(heads, model) -> dict:
    m = {}
    for li, h in heads:
        m.setdefault(li, torch.ones(model.cfg.n_head, device=model.tok_emb.weight.device))[h] = 0.0
    return m


@torch.no_grad()
def val_loss(model: GPT, data_dir: str, head_mask: dict | None) -> float:
    """Mean token loss over every full non-overlapping window of the validation split."""
    ids = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16).astype(np.int64)
    T = model.cfg.block_size
    n = (len(ids) - 1) // T
    x = torch.from_numpy(ids[: n * T]).view(n, T)
    y = torch.from_numpy(ids[1: n * T + 1]).view(n, T)
    for i, a in enumerate(b.attn for b in model.blocks):
        a.head_mask = None if head_mask is None or i not in head_mask else head_mask[i]
    dev = model.tok_emb.weight.device
    tot = 0.0
    for i in range(0, n, 16):
        xb = x[i:i + 16].to(dev)
        tot += model(xb, y[i:i + 16].to(dev))[1].item() * xb.size(0)
    for b in model.blocks:
        b.attn.head_mask = None
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--snapshots", default=None)
    ap.add_argument("--history", default=None)
    ap.add_argument("--L", type=int, default=100, help="length of the random sequence that is repeated")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--controls", type=int, default=20, help="random k-head ablations for the control")
    ap.add_argument("--out_json", default="runs/interpretability.json")
    ap.add_argument("--out_png", default="assets/induction_heads.png")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device

    model = load(a.ckpt, device)
    nl, nh = model.cfg.n_layer, model.cfg.n_head
    pool = token_pool(a.data_dir)
    seq = repeated_batch(pool, a.L, a.batch, seed=0, device=device)
    base = measure(model, seq, a.L)
    order = np.dstack(np.unravel_index(np.argsort(-base["induction"], axis=None), base["induction"].shape))[0]
    top = [tuple(map(int, x)) for x in order[: a.topk]]

    # causal test on fresh sequences (different seed than the one used to select heads)
    seq2 = repeated_batch(pool, a.L, a.batch, seed=1, device=device)
    clean = measure(model, seq2, a.L)
    abl = measure(model, seq2, a.L, mask_for(top, model))
    others = [(li, h) for li in range(nl) for h in range(nh) if (li, h) not in top]
    rng = random.Random(0)
    ctrl_second, ctrl_val = [], []
    clean_val = val_loss(model, a.data_dir, None)
    abl_val = val_loss(model, a.data_dir, mask_for(top, model))
    for _ in range(a.controls):
        heads = rng.sample(others, a.topk)
        ctrl_second.append(measure(model, seq2, a.L, mask_for(heads, model))["loss_second_half"])
        ctrl_val.append(val_loss(model, a.data_dir, mask_for(heads, model)))

    # same probe on natural text: random L-token passages from the val split, each repeated twice
    nat = natural_batch(a.data_dir, a.L, a.batch, seed=2, device=device)
    nat_clean = measure(model, nat, a.L)
    nat_abl = measure(model, nat, a.L, mask_for(top, model))
    nat_ctrl = [measure(model, nat, a.L, mask_for(rng.sample(others, a.topk), model))["loss_second_half"]
                for _ in range(a.controls)]

    result = {
        "natural_text_repeat": {
            "loss_first_half": nat_clean["loss_first_half"], "loss_second_half": nat_clean["loss_second_half"],
            "max_induction_score": float(nat_clean["induction"].max()),
            "second_half_loss_top_induction_ablated": nat_abl["loss_second_half"],
            "second_half_loss_random_ablated_mean": float(np.mean(nat_ctrl)),
        },
        "checkpoint": a.ckpt, "token_pool_size": len(pool), "L": a.L, "batch": a.batch, "uniform_loss_over_pool": float(np.log(len(pool))),
        "loss_first_half": clean["loss_first_half"], "loss_second_half": clean["loss_second_half"],
        "induction_scores": base["induction"].round(4).tolist(),
        "prev_token_scores": base["prev_token"].round(4).tolist(),
        "top_induction_heads": [{"layer": l, "head": h, "score": float(base["induction"][l, h])} for l, h in top],
        "ablation": {
            "k": a.topk,
            "second_half_loss": {"clean": clean["loss_second_half"], "top_induction_ablated": abl["loss_second_half"],
                                 "random_ablated_mean": float(np.mean(ctrl_second)),
                                 "random_ablated_max": float(np.max(ctrl_second))},
            "val_loss": {"clean": clean_val, "top_induction_ablated": abl_val,
                         "random_ablated_mean": float(np.mean(ctrl_val)), "random_ablated_max": float(np.max(ctrl_val))},
        },
    }

    dyn = None
    if a.snapshots:
        dyn = []
        for p in sorted(glob.glob(os.path.join(a.snapshots, "ckpt_*.pt"))):
            m = load(p, device)
            r = measure(m, seq, a.L)
            dyn.append({"iter": int(os.path.basename(p)[5:10]), "max_induction": float(r["induction"].max()),
                        "max_prev_token": float(r["prev_token"].max()),
                        "loss_first_half": r["loss_first_half"], "loss_second_half": r["loss_second_half"]})
        result["dynamics"] = dyn
        if a.history:
            with open(a.history) as f:
                result["dynamics_val_curve"] = [{"iter": e["iter"], "val": e["val"]} for e in json.load(f)["evals"]]

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ("induction_scores", "prev_token_scores",
                                                                     "dynamics", "dynamics_val_curve")}, indent=1))
    plot(result, a.out_png)


def plot(r: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_dyn = "dynamics" in r
    fig, axes = plt.subplots(1, 4 if has_dyn else 3, figsize=(19 if has_dyn else 14, 4.0), dpi=150)
    for ax, key, title in [(axes[0], "induction_scores", "Induction score"),
                           (axes[1], "prev_token_scores", "Previous-token score")]:
        M = np.array(r[key])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(title)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)

    ab = r["ablation"]["second_half_loss"]
    ax = axes[2]
    labels = ["clean", f"top-{r['ablation']['k']}\ninduction\nheads off", f"{r['ablation']['k']} random\nheads off\n(mean)"]
    vals = [ab["clean"], ab["top_induction_ablated"], ab["random_ablated_mean"]]
    ax.bar(labels, vals, color=["#2563eb", "#dc2626", "#9ca3af"])
    ax.axhline(r["loss_first_half"], color="black", ls="--", lw=1, label="1st half (no copying possible)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("loss on repeated half (nats/token)")
    ax.set_title("Causal test: in-context copying")
    ax.legend(frameon=False, fontsize=7)

    if has_dyn:
        ax = axes[3]
        d = r["dynamics"]
        it = [x["iter"] for x in d]
        ax.plot(it, [x["max_induction"] for x in d], "o-", ms=3, color="#7c3aed", label="max induction score")
        ax.plot(it, [x["max_prev_token"] for x in d], "o-", ms=3, color="#059669", label="max prev-token score")
        ax.set_xlabel("training iteration")
        ax.set_ylabel("attention score")
        ax.set_ylim(0, 1.05)
        ax2 = ax.twinx()
        ax2.plot(it, [x["loss_second_half"] for x in d], "s--", ms=3, color="#dc2626", label="repeated-half loss")
        ax2.set_ylabel("repeated-half loss", color="#dc2626")
        ax.set_title("Training dynamics (snapshots)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=True, framealpha=0.9, fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=3)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
