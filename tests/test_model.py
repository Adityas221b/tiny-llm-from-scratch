import dataclasses

import pytest
import torch

from tinyllm.model import GPT, GPTConfig, CausalSelfAttention, MLP, RMSNorm, apply_rope, rope_cache

BASE = dict(vocab_size=300, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
CFG = GPTConfig.preset("gpt2", **BASE)

# every (architecture, attention implementation) combination
VARIANTS = [(arch, sdpa) for arch in ("gpt2", "llama") for sdpa in (False, True)]
IDS = [f"{a}-{'sdpa' if s else 'manual'}" for a, s in VARIANTS]


def make(arch="gpt2", sdpa=False, **kw):
    return GPTConfig.preset(arch, **{**BASE, "use_sdpa": sdpa, **kw})


# --------------------------------------------------------------------------- shapes / basics

@pytest.mark.parametrize("arch,sdpa", VARIANTS, ids=IDS)
def test_forward_shapes(arch, sdpa):
    cfg = make(arch, sdpa)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_attention_shape():
    attn = CausalSelfAttention(CFG)
    x = torch.randn(2, 7, CFG.n_embd)
    y, (k, v) = attn(x)
    assert y.shape == x.shape
    assert k.shape == v.shape == (2, CFG.n_head, 7, CFG.n_embd // CFG.n_head)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_initial_loss_near_uniform(arch):
    torch.manual_seed(0)
    model = GPT(make(arch))
    idx = torch.randint(0, BASE["vocab_size"], (4, BASE["block_size"]))
    # independent targets: with tied weights, targets == inputs would be predicted better than chance
    tgt = torch.randint(0, BASE["vocab_size"], (4, BASE["block_size"]))
    _, loss = model(idx, tgt)
    assert abs(loss.item() - torch.log(torch.tensor(300.0)).item()) < 0.5


def test_weight_tying():
    model = GPT(CFG)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


# --------------------------------------------------------------------------- causality

@pytest.mark.parametrize("arch,sdpa", VARIANTS, ids=IDS)
def test_no_future_token_leakage(arch, sdpa):
    """Changing token t must not change logits at any position < t."""
    torch.manual_seed(0)
    cfg = make(arch, sdpa)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    t = 9
    idx2 = idx.clone()
    idx2[0, t:] = (idx2[0, t:] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(idx2)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-5)
    assert not torch.allclose(a[:, t:], b[:, t:])  # sanity: future positions do change


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_gradient_does_not_flow_from_future(arch):
    """d(output at position t) / d(input at position > t) must be exactly zero."""
    torch.manual_seed(0)
    cfg = make(arch)
    attn = CausalSelfAttention(cfg).eval()
    rope = rope_cache(cfg.block_size, cfg.n_embd // cfg.n_head, cfg.rope_base) if arch == "llama" else None
    x = torch.randn(1, 8, cfg.n_embd, requires_grad=True)
    attn(x, rope)[0][0, 3].sum().backward()
    assert torch.all(x.grad[0, 4:] == 0)
    assert torch.any(x.grad[0, :4] != 0)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_sdpa_matches_reference_attention(arch):
    """The fused path must equal the hand-written masked-softmax path."""
    torch.manual_seed(0)
    ref = GPT(make(arch, False)).eval()
    fused = GPT(make(arch, True)).eval()
    fused.load_state_dict(ref.state_dict())
    idx = torch.randint(0, 300, (2, BASE["block_size"]))
    with torch.no_grad():
        assert torch.allclose(ref(idx)[0], fused(idx)[0], atol=1e-5)


# --------------------------------------------------------------------------- KV cache

@pytest.mark.parametrize("arch,sdpa", VARIANTS, ids=IDS)
def test_kv_cache_matches_full_forward(arch, sdpa):
    """Prefill + one-token decode steps must give the same logits as a full forward pass."""
    torch.manual_seed(0)
    cfg = make(arch, sdpa)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    with torch.no_grad():
        full, _ = model(idx)
        cache: list = []
        prefill, _ = model(idx[:, :5], kv_cache=cache, start_pos=0)
        steps = [prefill]
        for t in range(5, cfg.block_size):
            out, _ = model(idx[:, t:t + 1], kv_cache=cache, start_pos=t)
            steps.append(out)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)
    assert cache[0][0].shape[2] == cfg.block_size


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_cached_generation_equals_uncached_greedy(arch):
    torch.manual_seed(0)
    model = GPT(make(arch, True)).eval()
    idx = torch.randint(0, 300, (2, 3))
    a = model.generate(idx, max_new_tokens=12, temperature=0, use_cache=True)   # stays within block_size
    b = model.generate(idx, max_new_tokens=12, temperature=0, use_cache=False)
    assert torch.equal(a, b)


@pytest.mark.parametrize("use_cache", [True, False])
def test_generate_beyond_context_and_topk(use_cache):
    torch.manual_seed(0)
    model = GPT(make("llama", True)).eval()
    idx = torch.zeros((2, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=40, temperature=0.9, top_k=5, use_cache=use_cache)
    assert out.shape == (2, 41)
    assert out.max() < 300


# --------------------------------------------------------------------------- components

def test_rope_is_relative():
    """<R_m q, R_n k> depends only on m - n."""
    torch.manual_seed(0)
    hd = 16
    cos, sin = rope_cache(64, hd, 10000.0)
    q, k = torch.randn(hd), torch.randn(hd)

    def score(m, n):
        qm = apply_rope(q.view(1, 1, 1, hd), cos[m:m + 1], sin[m:m + 1]).flatten()
        kn = apply_rope(k.view(1, 1, 1, hd), cos[n:n + 1], sin[n:n + 1]).flatten()
        return (qm @ kn).item()

    assert abs(score(5, 2) - score(40, 37)) < 1e-4
    assert abs(score(5, 2) - score(5, 4)) > 1e-3
    # rotation preserves norm
    r = apply_rope(q.view(1, 1, 1, hd), cos[7:8], sin[7:8]).flatten()
    assert torch.allclose(r.norm(), q.norm(), atol=1e-5)


def test_rmsnorm_unit_rms():
    x = torch.randn(4, 10, 32) * 7 + 3
    y = RMSNorm(32)(x)
    assert torch.allclose(y.pow(2).mean(-1).sqrt(), torch.ones(4, 10), atol=1e-3)


def test_swiglu_is_param_matched():
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    gelu = MLP(dataclasses.replace(CFG, n_embd=256, bias=False))
    swiglu = MLP(dataclasses.replace(CFG, n_embd=256, bias=False, mlp="swiglu"))
    assert abs(n(swiglu) - n(gelu)) / n(gelu) < 0.02


def test_head_mask_ablates_heads():
    torch.manual_seed(0)
    attn = CausalSelfAttention(make("llama")).eval()
    x = torch.randn(1, 6, BASE["n_embd"])
    attn.head_mask = torch.zeros(BASE["n_head"])
    y, _ = attn(x)
    assert torch.allclose(y, torch.zeros_like(y))   # no bias in llama preset -> exactly zero
    attn.head_mask = torch.ones(BASE["n_head"])
    attn.store_attn = True
    attn(x)
    a = attn.last_attn
    assert a.shape == (1, BASE["n_head"], 6, 6)
    assert torch.allclose(a.sum(-1), torch.ones(1, BASE["n_head"], 6))
    assert torch.all(a.triu(1) == 0)


# --------------------------------------------------------------------------- learning

@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_overfits_single_batch(arch):
    """A correct model should drive loss down quickly on one repeated batch."""
    torch.manual_seed(0)
    model = GPT(make(arch))
    opt = model.configure_optimizer(lr=3e-3, weight_decay=0.0)
    idx = torch.randint(0, 300, (4, BASE["block_size"] + 1))
    x, y = idx[:, :-1], idx[:, 1:]
    first = None
    for _ in range(100):
        _, loss = model(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first
