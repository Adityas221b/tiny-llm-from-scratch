"""Regenerate every results table in README.md from the measured run files in runs/.

Each table lives between `<!-- BEGIN:name -->` and `<!-- END:name -->` markers, so no
result in the README is typed by hand: re-run an experiment, then run this script.

    python make_report.py           # rewrite the tables in README.md
    python make_report.py --check   # exit 1 if README.md is out of date (used by the tests)
"""
import argparse
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(ROOT, "README.md")


def load(rel: str) -> dict:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def n_params(model_cfg: dict) -> int:
    """Exact parameter count of a run, rebuilt from the config stored in its history."""
    from tinyllm.model import GPT, GPTConfig
    return GPT(GPTConfig(**model_cfg)).num_params(non_embedding=False)


def fmt_m(n: int) -> str:
    return f"{n / 1e6:.2f}M"


# --------------------------------------------------------------------------- tables

def headline() -> str:
    rows = [
        ("**Jeeves & Wooster** (P.G. Wodehouse, 1.46M chars)", "**GPT-2 + RoPE**",
         "runs/jeeves_gpt2rope_eval_val.json", "runs/jeeves_gpt2rope_history.json", True),
        ("Jeeves & Wooster", "full Llama preset (RoPE + RMSNorm + SwiGLU)",
         "runs/jeeves_eval_val.json", "runs/jeeves_history.json", False),
        ("TinyShakespeare (1.1M chars)", "GPT-2-style",
         "runs/run2_eval_val.json", "runs/run2_1500iters.json", False),
    ]
    out = ["| corpus | model | held-out perplexity | bits / char | train time |", "|---|---|---|---|---|"]
    for corpus, model, ev_path, hist_path, bold in rows:
        ev, h = load(ev_path), load(hist_path)
        b = "**" if bold else ""
        out.append(f"| {corpus} | {model}, {fmt_m(n_params(h['model']))} params | "
                   f"{b}{ev['perplexity']:.2f}{b} | {b}{ev['bits_per_char']:.2f}{b} | "
                   f"{h['total_time_s'] / 60:.1f} min |")
    return "\n".join(out)


def ablation() -> str:
    s = load("runs/ablation/summary.json")
    res = s["results"]
    names = list(res)
    best = min(names, key=lambda n: res[n]["val_loss_mean"])
    base = res[names[0]]["val_loss_mean"]
    out = [f"| variant | val loss (mean ± std, {len(s['seeds'])} seeds) | val ppl | bits/char | Δ vs gpt2 | "
           "best iter (mean) | train time |", "|---|---|---|---|---|---|---|"]
    for i, n in enumerate(names):
        r = res[n]
        label = n.replace("llama (RoPE, RMSNorm, SwiGLU)", "llama (RoPE, RMSNorm, SwiGLU, no biases)")
        delta = "—" if i == 0 else f"{r['val_loss_mean'] - base:+.3f}".replace("-", "−")
        cells = [label, f"{r['val_loss_mean']:.3f} ± {r['val_loss_std']:.3f}", f"{r['val_ppl_at_mean']:.2f}",
                 f"{r['bpc_mean']:.3f}", delta]
        if n == best:
            cells = [f"**{c}**" for c in cells]
        cells += [f"{r['best_iter_mean']:.0f}", f"{r['time_s_mean'] / 60:.1f} min"]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def interpretability() -> str:
    r = load("runs/jeeves_interpretability.json")
    d = r["dynamics"]
    first, last = d[0], d[-1]
    peak = max(d, key=lambda x: x["max_prev_token"])
    formed = next(x for x in d if x["max_prev_token"] >= 0.95 * peak["max_prev_token"])  # when the head forms
    nat, ab = r["natural_text_repeat"], r["ablation"]
    sh, vl = ab["second_half_loss"], ab["val_loss"]
    rows = [
        ("max previous-token score",
         f"{first['max_prev_token']:.3f} at iteration {first['iter']} → **{formed['max_prev_token']:.3f} by "
         f"iteration {formed['iter']}** (peak {peak['max_prev_token']:.3f} at {peak['iter']}), "
         f"{last['max_prev_token']:.3f} at iteration {last['iter']}"),
        ("max induction score",
         f"**≤ {max(x['max_induction'] for x in d):.3f} at every snapshot** "
         # uniform attention: query i attends to i+1 keys; averaged over the second-half queries
         f"(chance ≈ {sum(1 / (i + 1) for i in range(r['L'], 2 * r['L'] - 1)) / (r['L'] - 1):.3f})"),
        ("loss on repeated random tokens, first half → second half",
         f"{r['loss_first_half']:.2f} → {r['loss_second_half']:.2f}"
         + (" (no gain from having seen the sequence)" if r["loss_second_half"] >= r["loss_first_half"] else "")),
        ("loss on a repeated natural Jeeves passage, first → second",
         f"{nat['loss_first_half']:.2f} → {nat['loss_second_half']:.2f}"
         + (" (no gain)" if nat["loss_second_half"] >= nat["loss_first_half"] else "")),
        (f"repeated-half loss: clean / top-{ab['k']} \"induction\" heads off / {ab['k']} random heads off",
         f"{sh['clean']:.2f} / {sh['top_induction_ablated']:.2f} / {sh['random_ablated_mean']:.2f}"),
        (f"val loss: clean / top-{ab['k']} heads off / {ab['k']} random heads off",
         f"{vl['clean']:.3f} / {vl['top_induction_ablated']:.3f} / {vl['random_ablated_mean']:.3f}"),
    ]
    return "\n".join(["| measurement | value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows])


def kv_cache() -> str:
    r = load("runs/kv_cache_bench.json")["results"]
    best = max((x for x in r), key=lambda x: x["speedup"])
    out = ["| device | tokens generated | no cache | KV cache | speedup |", "|---|---|---|---|---|"]
    for x in r:
        dev = "CPU" if x["device"] == "cpu" else "GPU (RTX 5050 laptop)"
        sp = f"{x['speedup']:.1f}×"
        out.append(f"| {dev} | {x['new_tokens']} | {x['tok_per_s_no_cache']:.1f} tok/s | "
                   f"{x['tok_per_s_kv_cache']:.1f} tok/s | {'**' + sp + '**' if x is best else sp} |")
    # CPU rows first, as in the write-up
    return "\n".join(out[:2] + sorted(out[2:], key=lambda l: not l.startswith("| CPU")))


TABLES = {"headline": headline, "ablation": ablation, "interpretability": interpretability, "kv_cache": kv_cache}


def render(text: str) -> str:
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README.md is missing the <!-- BEGIN:{name} --> / <!-- END:{name} --> markers")
        body = fn()
        text = pat.sub(lambda m: m.group(1) + body + m.group(2), text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if README.md does not match runs/")
    a = ap.parse_args()
    with open(README, encoding="utf-8") as f:
        current = f.read()
    new = render(current)
    if a.check:
        if new != current:
            print("README.md results tables are out of date: run `python make_report.py`")
            sys.exit(1)
        print("README.md results tables match runs/")
        return
    with open(README, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    print("updated" if new != current else "already up to date", "README.md:", ", ".join(TABLES))


if __name__ == "__main__":
    main()
