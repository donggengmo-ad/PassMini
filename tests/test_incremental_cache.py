import torch

from scripts.models import AutoregressiveTCN, AutoregressiveTransformer
from scripts.tokenizer import CharTokenizer


def _assert_incremental_logits_match_full_forward(model, tokenizer):
    """逐步消费前缀，并与完整因果前向传播的最后位置比较。"""

    model.eval()
    token_ids = tokenizer.encode("abc")
    prefix = [tokenizer.bos_id]
    with torch.inference_mode():
        state = model.initial_state(batch_size=1)
        torch.testing.assert_close(
            state.next_logits,
            model(torch.tensor([prefix]))[:, -1, :],
            atol=1e-6,
            rtol=1e-5,
        )
        for token_id in token_ids:
            prefix.append(token_id)
            state = model.advance_state(state, torch.tensor([token_id]))
            torch.testing.assert_close(
                state.next_logits,
                model(torch.tensor([prefix]))[:, -1, :],
                atol=1e-6,
                rtol=1e-5,
            )


def _clone_cache_tensors(cache):
    """递归复制 cache 中的张量，用于检测父状态是否被原地修改。"""

    if isinstance(cache, torch.Tensor):
        return cache.clone()
    if isinstance(cache, tuple):
        return tuple(_clone_cache_tensors(value) for value in cache)
    if hasattr(cache, "__dataclass_fields__"):
        return tuple(
            _clone_cache_tensors(getattr(cache, name))
            for name in cache.__dataclass_fields__
        )
    return cache


def _assert_cache_tensors_equal(cache, snapshot):
    if isinstance(cache, torch.Tensor):
        torch.testing.assert_close(cache, snapshot)
        return
    if isinstance(cache, tuple):
        for value, expected in zip(cache, snapshot, strict=True):
            _assert_cache_tensors_equal(value, expected)
        return
    if hasattr(cache, "__dataclass_fields__"):
        for name, expected in zip(cache.__dataclass_fields__, snapshot, strict=True):
            _assert_cache_tensors_equal(getattr(cache, name), expected)


def test_tcn_uses_weight_tying_and_incremental_layer_cache():
    tokenizer = CharTokenizer.from_text(["abc"])
    model = AutoregressiveTCN(
        tokenizer,
        embedding_dim=4,
        channels=6,
        kernel_size=3,
        dilations=(1, 2),
    )

    assert model.output_projection.out_features == model.embedding_dim
    assert not hasattr(model, "vocab_projection")
    _assert_incremental_logits_match_full_forward(model, tokenizer)

    state = model.initial_state(1)
    snapshot = _clone_cache_tensors(state.cache)
    model.advance_state(state, torch.tensor([tokenizer.token_to_id["a"]]))
    _assert_cache_tensors_equal(state.cache, snapshot)

    histories = state.cache.layer_inputs
    assert [history.shape[-1] for history in histories] == [2, 4]


def test_transformer_uses_weight_tying_and_incremental_kv_cache():
    tokenizer = CharTokenizer.from_text(["abc"])
    model = AutoregressiveTransformer(
        tokenizer,
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=12,
        max_length=8,
    )

    assert not hasattr(model, "output_projection")
    _assert_incremental_logits_match_full_forward(model, tokenizer)

    state = model.initial_state(1)
    snapshot = _clone_cache_tensors(state.cache)
    child = model.advance_state(state, torch.tensor([tokenizer.token_to_id["a"]]))
    _assert_cache_tensors_equal(state.cache, snapshot)

    for parent_layer, child_layer in zip(
        state.cache.layers,
        child.cache.layers,
        strict=True,
    ):
        assert parent_layer.keys.shape == (1, model.nhead, 1, model.head_dim)
        assert parent_layer.values.shape == (1, model.nhead, 1, model.head_dim)
        assert child_layer.keys.shape[-2] == 2
        assert child_layer.values.shape[-2] == 2


def test_tied_output_keeps_pad_embedding_gradient_zero():
    tokenizer = CharTokenizer.from_text(["abc"])
    models = [
        AutoregressiveTCN(tokenizer, embedding_dim=4, channels=6),
        AutoregressiveTransformer(tokenizer, d_model=8, nhead=2, num_layers=1),
    ]
    inputs = torch.tensor([[tokenizer.bos_id, tokenizer.token_to_id["a"]]])
    targets = torch.tensor([[tokenizer.token_to_id["a"], tokenizer.eos_id]])

    for model in models:
        model.zero_grad()
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        assert torch.all(model.embedding.weight.grad[tokenizer.pad_id] == 0)
