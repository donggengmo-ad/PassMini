import json
import math
from pathlib import Path

import torch

from scripts.experiment import model_config_from_dict
from scripts.models import AutoregressiveMLP
from scripts.pipeline import build_model
from scripts.tokenizer import CharTokenizer


def test_mlp_forward_vectorizes_left_padded_context_windows():
    tokenizer = CharTokenizer.from_text(["abc"])
    model = AutoregressiveMLP(
        tokenizer,
        tau=3,
        embedding_dim=4,
        hidden_size=6,
    )
    input_ids = torch.tensor(
        [[tokenizer.bos_id, tokenizer.token_to_id["a"], tokenizer.token_to_id["b"]]]
    )

    logits = model(input_ids)

    padded = torch.nn.functional.pad(input_ids, (model.tau - 1, 0), value=model.pad_id)
    windows = padded.unfold(1, model.tau, 1)
    embeddings = model.embedding(windows).flatten(start_dim=2)
    expected = model.output_projection(
        torch.nn.functional.gelu(model.input_projection(embeddings))
    ) @ model.embedding.weight.T
    assert windows.tolist() == [
        [
            [tokenizer.pad_id, tokenizer.pad_id, tokenizer.bos_id],
            [tokenizer.pad_id, tokenizer.bos_id, tokenizer.token_to_id["a"]],
            [tokenizer.bos_id, tokenizer.token_to_id["a"], tokenizer.token_to_id["b"]],
        ]
    ]
    assert logits.shape == (1, 3, tokenizer.vocab_size)
    torch.testing.assert_close(logits, expected)


def test_mlp_incremental_batch_state_matches_full_forward():
    tokenizer = CharTokenizer.from_text(["abc"])
    model = AutoregressiveMLP(
        tokenizer,
        tau=3,
        embedding_dim=4,
        hidden_size=6,
    )
    model.eval()
    prefix = [tokenizer.bos_id]

    with torch.inference_mode():
        state = model.initial_state(2)
        repeated_prefix = torch.tensor([prefix, prefix])
        torch.testing.assert_close(
            state.next_logits,
            model(repeated_prefix)[:, -1, :],
        )
        for character in "abc":
            token_id = tokenizer.token_to_id[character]
            prefix.append(token_id)
            state = model.advance_state(state, torch.full((2,), token_id))
            repeated_prefix = torch.tensor([prefix, prefix])
            torch.testing.assert_close(
                state.next_logits,
                model(repeated_prefix)[:, -1, :],
            )
    assert state.cache.shape == (2, model.tau)


def test_mlp_batch_log_probabilities_use_one_forward_and_include_eos():
    tokenizer = CharTokenizer.from_text(["abc"])
    model = AutoregressiveMLP(
        tokenizer,
        tau=4,
        embedding_dim=4,
        hidden_size=6,
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    forward_calls = []
    handle = model.register_forward_hook(
        lambda _module, _inputs, _output: forward_calls.append(1)
    )

    texts = ["", "a", "abc"]
    scores = model.log_probabilities(texts)
    handle.remove()
    expected = torch.tensor(
        [-(len(text) + 1) * math.log(model.vocab_size) for text in texts]
    )

    assert len(forward_calls) == 1
    torch.testing.assert_close(scores, expected)


def test_mlp_uses_weight_tying_and_keeps_pad_gradient_zero():
    tokenizer = CharTokenizer.from_text(["abc"])
    model = AutoregressiveMLP(
        tokenizer,
        tau=2,
        embedding_dim=4,
        hidden_size=6,
    )
    inputs = torch.tensor([[tokenizer.bos_id, tokenizer.token_to_id["a"]]])
    targets = torch.tensor([[tokenizer.token_to_id["a"], tokenizer.eos_id]])

    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, tokenizer.vocab_size),
        targets.reshape(-1),
    )
    loss.backward()

    assert model.input_projection.in_features == model.tau * model.embedding_dim
    assert model.output_projection.out_features == model.embedding_dim
    assert not hasattr(model, "vocab_projection")
    assert torch.all(model.embedding.weight.grad[tokenizer.pad_id] == 0)


def test_mlp_tier_parameter_counts_match_declared_configs():
    root = Path(__file__).resolve().parents[1]
    tokenizer = CharTokenizer.from_json(
        root / "data" / "processed" / "tokenizer.json"
    )
    tiers = json.loads(
        (root / "output" / "model_tiers.json").read_text(encoding="utf-8")
    )

    tier_names = ("low", "medium", "high")
    assert [tiers[tier]["mlp"]["tau"] for tier in tier_names] == [1, 4, 8]
    assert [tiers[tier]["mlp"]["embedding_dim"] for tier in tier_names] == [
        64,
        128,
        256,
    ]
    assert [tiers[tier]["mlp"]["hidden_size"] for tier in tier_names] == [
        128,
        256,
        512,
    ]
    for tier in tier_names:
        declared = tiers[tier]["mlp"]
        structure = {
            key: declared[key]
            for key in ("model_type", "tau", "embedding_dim", "hidden_size")
        }
        config = model_config_from_dict(structure)
        model = build_model(config, tokenizer)
        actual = sum(parameter.numel() for parameter in model.parameters())
        assert actual == declared["parameter_count"]
