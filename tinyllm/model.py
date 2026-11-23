"""A small decoder-only transformer, written from scratch in PyTorch.

Every architectural choice that differs between GPT-2 and Llama-style models is a config switch,
so the two can be compared under an identical training budget (see ablate.py):

    component        GPT-2 style (baseline)        Llama style
    ---------        ----------------------        -----------
    positions        learned absolute embedding    rotary (RoPE) applied to q, k
    normalisation    LayerNorm                     RMSNorm
    MLP              Linear-GELU-Linear (4x)       SwiGLU (gated, ~8/3x, param-matched)

Also: weight tying, explicit or fused (SDPA) causal attention, a KV cache for O(1)-per-token
decoding, and hooks for interpretability (store attention maps, ablate individual heads).
Only torch.nn primitives (Linear, Embedding, Dropout, LayerNorm) are used from the library.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 128          # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
    use_sdpa: bool = False         # False = explicit masked softmax; True = F.scaled_dot_product_attention
    pos_emb: str = "learned"       # "learned" | "rope"
    norm: str = "layernorm"        # "layernorm" | "rmsnorm"
    mlp: str = "gelu"              # "gelu" | "swiglu"
    rope_base: float = 10000.0

    @classmethod
    def preset(cls, name: str, **kw) -> "GPTConfig":
        presets = {
            "gpt2": dict(pos_emb="learned", norm="layernorm", mlp="gelu"),
            "llama": dict(pos_emb="rope", norm="rmsnorm", mlp="swiglu", bias=False),
        }
        return cls(**{**presets[name], **kw})


# --------------------------------------------------------------------------- building blocks

class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * g  -- no mean-centering, no bias (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def make_norm(cfg: GPTConfig) -> nn.Module:
    return RMSNorm(cfg.n_embd) if cfg.norm == "rmsnorm" else nn.LayerNorm(cfg.n_embd, bias=cfg.bias)


def rope_cache(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (seq_len, head_dim/2) for rotary position embeddings (Su et al., 2021)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs (x0,x1),(x2,x3),... of the head dimension by position-dependent angles.
    x: (B, h, T, hd); cos/sin: (T, hd/2)."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


# --------------------------------------------------------------------------- attention

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t can only attend to positions <= t."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        self.use_sdpa = cfg.use_sdpa
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
        # interpretability hooks
        self.store_attn = False
        self.last_attn: torch.Tensor | None = None
        self.head_mask: torch.Tensor | None = None   # (n_head,) multipliers; 0 ablates a head

    def forward(self, x: torch.Tensor, rope: tuple | None = None, kv: tuple | None = None,
                start_pos: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, h, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if rope is not None:
            cos, sin = rope[0][start_pos:start_pos + T], rope[1][start_pos:start_pos + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv is not None:                       # append to the KV cache
            k = torch.cat([kv[0], k], dim=2)
            v = torch.cat([kv[1], v], dim=2)
        new_kv = (k, v)
        S = k.size(2)                            # total key length = start_pos + T

        if self.use_sdpa and not self.store_attn:
            # is_causal aligns the mask top-left, which is only correct when q and k have equal length;
            # a single decode step (T == 1) may attend to every cached key, so no mask is needed.
            assert T == S or T == 1, "chunked prefill with a cache is not supported on the SDPA path"
            y = F.scaled_dot_product_attention(q, k, v, is_causal=(T == S and T > 1),
                                               dropout_p=self.dropout if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)          # (B, h, T, S)
            att = att.masked_fill(~self.mask[start_pos:start_pos + T, :S], float("-inf"))
            att = F.softmax(att.float(), dim=-1).type_as(q)
            if self.store_attn:
                self.last_attn = att.detach()
            y = self.attn_drop(att) @ v                                          # (B, h, T, hd)

        if self.head_mask is not None:
            y = y * self.head_mask.view(1, -1, 1, 1).to(y.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), new_kv


# --------------------------------------------------------------------------- MLP / block / model

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.kind = cfg.mlp
        if cfg.mlp == "swiglu":
            # 3 matrices of d x h with h = 8/3 d keeps the parameter count equal to a 4x GELU MLP
            hidden = int(round(8 * cfg.n_embd / 3 / 16)) * 16
            self.fc = nn.Linear(cfg.n_embd, 2 * hidden, bias=cfg.bias)   # gate and up, fused
        else:
            hidden = 4 * cfg.n_embd
            self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            gate, up = self.fc(x).chunk(2, dim=-1)
            h = F.silu(gate) * up
        else:
            h = F.gelu(self.fc(x))
        return self.drop(self.proj(h))


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(Norm(x)), then x + MLP(Norm(x))."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None, kv=None, start_pos=0):
        a, new_kv = self.attn(self.ln1(x), rope, kv, start_pos)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos_emb == "learned" else None
        if cfg.pos_emb == "rope":
            cos, sin = rope_cache(cfg.block_size, cfg.n_embd // cfg.n_head, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                 # weight tying

        self.apply(self._init_weights)
        # GPT-2: scale residual projections by 1/sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if non_embedding and self.pos_emb is not None:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                kv_cache: list | None = None, start_pos: int = 0):
        """If kv_cache is a list (empty on the first call) it is filled/extended in place."""
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size, \
            f"positions {start_pos}+{T} exceed block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(start_pos, start_pos + T, device=idx.device))
        x = self.drop(x)
        rope = (self.rope_cos, self.rope_sin) if self.cfg.pos_emb == "rope" else None
        for i, block in enumerate(self.blocks):
            past = kv_cache[i] if kv_cache is not None and i < len(kv_cache) else None
            x, new_kv = block(x, rope, past, start_pos)
            if kv_cache is not None:
                if i < len(kv_cache):
                    kv_cache[i] = new_kv
                else:
                    kv_cache.append(new_kv)
        logits = self.lm_head(self.ln_f(x))                       # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        # decay 2D weights (matmuls, embeddings); no decay for biases and norm gains
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        return torch.multinomial(F.softmax(logits.float(), dim=-1), num_samples=1)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, use_cache: bool = True) -> torch.Tensor:
        """Autoregressive sampling. With use_cache, each new token costs one single-position forward
        pass over cached keys/values instead of re-running the whole context. When the context fills
        up, the cache is rebuilt from the most recent block_size // 2 tokens."""
        if not use_cache:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.cfg.block_size:])
                idx = torch.cat([idx, self._sample(logits[:, -1, :], temperature, top_k)], dim=1)
            return idx

        cache: list = []
        ctx = idx[:, -self.cfg.block_size:]
        logits, _ = self(ctx, kv_cache=cache, start_pos=0)                 # prefill
        pos = ctx.size(1)
        for _ in range(max_new_tokens):
            next_id = self._sample(logits[:, -1, :], temperature, top_k)
            idx = torch.cat([idx, next_id], dim=1)
            if pos >= self.cfg.block_size:                                # context full: re-prefill
                cache = []
                ctx = idx[:, -(self.cfg.block_size // 2):]
                logits, _ = self(ctx, kv_cache=cache, start_pos=0)
                pos = ctx.size(1)
            else:
                logits, _ = self(next_id, kv_cache=cache, start_pos=pos)  # one-token decode step
                pos += 1
        return idx
