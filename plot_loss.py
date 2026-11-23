"""Plot loss curves from training history JSON files.

    python plot_loss.py --history out/history.json --out assets/loss_curve.png
    python plot_loss.py --history runs/run2_1500iters.json --compare runs/run1_5000iters_overfit.json \
        --out assets/loss_curve.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(p):
    with open(p) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="out/history.json")
    ap.add_argument("--compare", default=None, help="optional second run (e.g. the overfitting run)")
    ap.add_argument("--out", default="assets/loss_curve.png")
    ap.add_argument("--title", default="tiny-llm-from-scratch: TinyShakespeare, byte-level BPE (vocab 512), 4.9M params")
    a = ap.parse_args()

    h = load(a.history)
    ncols = 2 if a.compare else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.2), dpi=150, squeeze=False)

    ax = axes[0, 0]
    tc, ev = h["train_curve"], h["evals"]
    ax.plot([p["iter"] for p in tc], [p["loss"] for p in tc], color="#9aa5b1", lw=1, label="train (single batch)")
    ax.plot([e["iter"] for e in ev], [e["train"] for e in ev], "o-", color="#2563eb", ms=3, label="train (eval avg)")
    ax.plot([e["iter"] for e in ev], [e["val"] for e in ev], "o-", color="#dc2626", ms=3, label="val (eval avg)")
    ax.set_title(f"Final run: {h['config']['max_iters']} iters, cosine LR")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy (nats / BPE token)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    if a.compare:
        c = load(a.compare)
        ax = axes[0, 1]
        ce = c["evals"]
        ax.plot([e["iter"] for e in ce], [e["train"] for e in ce], "o-", color="#2563eb", ms=3, label="train")
        ax.plot([e["iter"] for e in ce], [e["val"] for e in ce], "o-", color="#dc2626", ms=3, label="val")
        best = min(ce, key=lambda e: e["val"])
        ax.axvline(best["iter"], color="#6b7280", ls="--", lw=1)
        ax.annotate(f"best val {best['val']:.2f} @ {best['iter']}", (best["iter"], best["val"]),
                    xytext=(10, 25), textcoords="offset points", fontsize=8)
        ax.set_title(f"Earlier run: {c['config']['max_iters']} iters (overfits)")
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle(a.title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
