import torch

from tinyllm.model import GPT, GPTConfig, CausalSelfAttention

CFG = GPTConfig(vocab_size=300, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def test_forward_shapes():
    model = GPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, CFG.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_attention_shape():
    attn = CausalSelfAttention(CFG)
    x = torch.randn(2, 7, CFG.n_embd)
    assert attn(x).shape == x.shape


def test_initial_loss_near_uniform():
    torch.manual_seed(0)
    model = GPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
    _, loss = model(idx, idx)
    expected = torch.log(torch.tensor(float(CFG.vocab_size)))
    assert abs(loss.item() - expected.item()) < 0.5


def test_no_future_token_leakage():
    """Changing token t must not change logits at any position < t."""
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.randint(0, CFG.vocab_size, (1, CFG.block_size))
    t = 9
    idx2 = idx.clone()
    idx2[0, t:] = (idx2[0, t:] + 1) % CFG.vocab_size
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(idx2)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6)
    assert not torch.allclose(a[:, t:], b[:, t:])  # sanity: future positions do change


def test_gradient_does_not_flow_from_future():
    """d(output at position t) / d(input at position > t) must be zero."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(CFG).eval()
    x = torch.randn(1, 8, CFG.n_embd, requires_grad=True)
    attn(x)[0, 3].sum().backward()
    assert torch.all(x.grad[0, 4:] == 0)
    assert torch.any(x.grad[0, :4] != 0)


def test_sdpa_matches_reference_attention():
    """The fused path must equal the hand-written masked-softmax path."""
    torch.manual_seed(0)
    ref = CausalSelfAttention(CFG).eval()
    fused = CausalSelfAttention(GPTConfig(**{**CFG.__dict__, "use_sdpa": True})).eval()
    fused.load_state_dict(ref.state_dict())
    x = torch.randn(2, CFG.block_size, CFG.n_embd)
    assert torch.allclose(ref(x), fused(x), atol=1e-5)


def test_no_future_token_leakage_sdpa():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**CFG.__dict__, "use_sdpa": True})).eval()
    idx = torch.randint(0, CFG.vocab_size, (1, CFG.block_size))
    idx2 = idx.clone()
    idx2[0, 9:] = (idx2[0, 9:] + 1) % CFG.vocab_size
    with torch.no_grad():
        assert torch.allclose(model(idx)[0][:, :9], model(idx2)[0][:, :9], atol=1e-6)


def test_weight_tying():
    model = GPT(CFG)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_generate_length_and_topk():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.zeros((2, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20, temperature=0.9, top_k=5)  # exceeds block_size: cropping path
    assert out.shape == (2, 21)
    assert out.max() < CFG.vocab_size


def test_overfits_single_batch():
    """A correct model should drive loss down quickly on one repeated batch."""
    torch.manual_seed(0)
    model = GPT(CFG)
    opt = model.configure_optimizer(lr=3e-3, weight_decay=0.0)
    idx = torch.randint(0, CFG.vocab_size, (4, CFG.block_size + 1))
    x, y = idx[:, :-1], idx[:, 1:]
    first = None
    for _ in range(100):
        _, loss = model(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first
