| variant | val loss (mean ± std) | val ppl | bits/char | Δ vs gpt2 | best iter | train min |
|---|---|---|---|---|---|---|
| gpt2 (learned pos, LayerNorm, GELU) | 2.793 ± 0.010 | 16.34 | 2.146 | +0.000 | 1433 | 2.0 |
| gpt2 + RoPE | 2.681 ± 0.012 | 14.59 | 2.059 | -0.113 | 1267 | 2.5 |
| gpt2 + RMSNorm + SwiGLU | 2.808 ± 0.013 | 16.58 | 2.157 | +0.015 | 1267 | 2.6 |
| llama (RoPE, RMSNorm, SwiGLU) | 2.736 ± 0.011 | 15.43 | 2.102 | -0.057 | 1200 | 2.9 |
