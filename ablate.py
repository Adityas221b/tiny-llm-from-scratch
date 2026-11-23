"""Architecture ablation: GPT-2 style -> Llama style, one component at a time, several seeds each.

Every run uses the identical data, tokenizer, model size, optimiser and step budget; only the
architecture switches differ. Each run's best-val checkpoint is scored on the *full* validation
split with evaluate.py, and results are reported as mean +- std over seeds.

    python ablate.py --seeds 1 2 3
Writes runs/ablation/summary.json, runs/ablation/summary.md and assets/ablation.png.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys

VARIANTS = {
    "gpt2 (learned pos, LayerNorm, GELU)": ["--arch", "gpt2"],
    "gpt2 + RoPE":                         ["--arch", "gpt2", "--pos_emb", "rope"],
    "gpt2 + RMSNorm + SwiGLU":             ["--arch", "gpt2", "--norm", "rmsnorm", "--mlp", "swiglu"],
    "llama (RoPE, RMSNorm, SwiGLU)":       ["--arch", "llama"],
}


def slug(name: str) -> str:
    return name.split(" (")[0].replace(" + ", "+").replace(" ", "_").lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--work_dir", default="ablation_ckpts")
    ap.add_argument("--max_iters", type=int, default=1500)
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    os.makedirs("runs/ablation", exist_ok=True)

    results = {}
    for name, flags in VARIANTS.items():
        results[name] = []
        for seed in a.seeds:
            out = os.path.join(a.work_dir, f"{slug(name)}_s{seed}")
            ev_path = os.path.join("runs/ablation", f"{slug(name)}_s{seed}_eval.json")
            hist_path = os.path.join("runs/ablation", f"{slug(name)}_s{seed}_history.json")
            if not (a.skip_existing and os.path.exists(ev_path)):
                print(f"\n=== {name} | seed {seed} ===", flush=True)
                subprocess.run([sys.executable, "-u", "train.py", *flags, "--seed", str(seed),
                                "--max_iters", str(a.max_iters), "--out_dir", out], check=True)
                subprocess.run([sys.executable, "evaluate.py", "--ckpt", os.path.join(out, "ckpt.pt"),
                                "--split", "val", "--out", ev_path], check=True, stdout=subprocess.DEVNULL)
                os.replace(os.path.join(out, "history.json"), hist_path)
            with open(ev_path) as f:
                ev = json.load(f)
            with open(hist_path) as f:
                h = json.load(f)
            results[name].append({"seed": seed, "val_loss": ev["loss_nats_per_token"], "val_ppl": ev["perplexity"],
                                  "bpc": ev["bits_per_char"], "best_iter": ev["checkpoint_iter"],
                                  "time_s": h["total_time_s"], "final_train_est": h["evals"][-1]["train"],
                                  "final_val_est": h["evals"][-1]["val"]})

    summary, rows = {}, []
    base = None
    for name, runs in results.items():
        losses = [r["val_loss"] for r in runs]
        m = statistics.mean(losses)
        s = statistics.stdev(losses) if len(losses) > 1 else 0.0
        base = m if base is None else base
        summary[name] = {"runs": runs, "val_loss_mean": m, "val_loss_std": s, "val_ppl_at_mean": math.exp(m),
                         "bpc_mean": statistics.mean(r["bpc"] for r in runs),
                         "best_iter_mean": statistics.mean(r["best_iter"] for r in runs),
                         "time_s_mean": statistics.mean(r["time_s"] for r in runs)}
        rows.append(f"| {name} | {m:.3f} ± {s:.3f} | {math.exp(m):.2f} | {summary[name]['bpc_mean']:.3f} | "
                    f"{m - base:+.3f} | {summary[name]['best_iter_mean']:.0f} | {summary[name]['time_s_mean'] / 60:.1f} |")

    with open("runs/ablation/summary.json", "w") as f:
        json.dump({"seeds": a.seeds, "max_iters": a.max_iters, "results": summary}, f, indent=1)
    table = ("| variant | val loss (mean ± std) | val ppl | bits/char | Δ vs gpt2 | best iter | train min |\n"
             "|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    with open("runs/ablation/summary.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)
    plot(results, a)


def plot(results, a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#6b7280", "#2563eb", "#d97706", "#dc2626"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), dpi=150)
    for c, (name, runs) in zip(colors, results.items()):
        for i, r in enumerate(runs):
            with open(os.path.join("runs/ablation", f"{slug(name)}_s{r['seed']}_history.json")) as f:
                ev = json.load(f)["evals"]
            ax1.plot([e["iter"] for e in ev], [e["val"] for e in ev], color=c, alpha=0.8, lw=1.2,
                     label=name if i == 0 else None)
        ys = [r["val_loss"] for r in runs]
        x = list(results).index(name)
        ax2.scatter([x] * len(ys), ys, color=c, zorder=3)
        ax2.errorbar([x], [statistics.mean(ys)], yerr=[statistics.stdev(ys) if len(ys) > 1 else 0],
                     fmt="_", color="black", capsize=8, ms=20, zorder=4)
    ax1.set_ylim(2.6, 4.0)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("val loss (eval estimate, nats/token)")
    ax1.set_title(f"Validation curves ({len(a.seeds)} seeds each)")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_xticks(range(len(results)), [n.split(" (")[0] for n in results], fontsize=8)
    ax2.set_ylabel("full-split val loss of best checkpoint")
    ax2.set_title("Best checkpoint, full validation split")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    os.makedirs("assets", exist_ok=True)
    fig.savefig("assets/ablation.png")
    print("saved assets/ablation.png")


if __name__ == "__main__":
    main()
