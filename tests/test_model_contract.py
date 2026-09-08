import inspect

import pytest
import torch

from scripts.models import (
    AutoregressiveBigram,
    AutoregressiveGRU,
    AutoregressiveMLP,
    AutoregressivePasswordModel,
    AutoregressiveTCN,
    AutoregressiveTransformer,
    PasswordModel,
)
from scripts.tokenizer import CharTokenizer


@pytest.fixture
def tokenizer():
    return CharTokenizer.from_text(["ab", "abc"])


@pytest.mark.parametrize(
    "factory",
    [
        lambda t: AutoregressiveBigram(t),
        lambda t: AutoregressiveMLP(t, tau=2, embedding_dim=4, hidden_size=5),
        lambda t: AutoregressiveGRU(t, embedding_dim=4, hidden_size=5),
        lambda t: AutoregressiveTCN(t, embedding_dim=4, channels=5),
        lambda t: AutoregressiveTransformer(t, d_model=4, nhead=2, num_layers=1),
    ],
)
def test_common_model_contract(factory, tokenizer):
    model = factory(tokenizer)
    assert isinstance(model, PasswordModel)
    assert isinstance(model, AutoregressivePasswordModel)
    assert model.pad_id == tokenizer.pad_id
    assert model.vocab_size == tokenizer.vocab_size
    generated = model.generate_batch(2, 4, generator=torch.Generator().manual_seed(1))
    assert len(generated) == 2
    scores = model.log_probabilities(["a", "ab"])
    assert scores.shape == (2,)
    torch.testing.assert_close(model.surprisal_bits_batch(["a", "ab"]), -scores / torch.log(torch.tensor(2.0)))


def test_generate_delegates_to_batch(tokenizer):
    model = AutoregressiveGRU(tokenizer, embedding_dim=4, hidden_size=5)
    generator = torch.Generator().manual_seed(42)
    expected = model.generate_batch(1, 5, generator=generator)[0]
    actual = model.generate(5, generator=torch.Generator().manual_seed(42))
    assert actual == expected


@pytest.mark.parametrize(
    "factory",
    [
        lambda t: AutoregressiveBigram(t),
        lambda t: AutoregressiveMLP(t, tau=2, embedding_dim=4, hidden_size=5),
        lambda t: AutoregressiveGRU(t, embedding_dim=4, hidden_size=5),
        lambda t: AutoregressiveTCN(t, embedding_dim=4, channels=5),
        lambda t: AutoregressiveTransformer(t, d_model=4, nhead=2, num_layers=1),
    ],
)
def test_state_selection_and_stacking_preserve_each_batch_row(factory, tokenizer):
    model = factory(tokenizer)
    model.eval()
    state = model.initial_state(3)
    indices = torch.tensor([2, 0, 2], device=state.next_logits.device)
    selected = model.select_state(state, indices)
    stacked = model.stack_states(
        [model.select_state(state, [2]), model.select_state(state, [0, 2])]
    )

    expected_logits = state.next_logits.index_select(0, indices)
    torch.testing.assert_close(selected.next_logits, expected_logits)
    torch.testing.assert_close(stacked.next_logits, expected_logits)

    token_ids = torch.tensor(
        [tokenizer.token_to_id["a"], tokenizer.token_to_id["b"], tokenizer.eos_id],
        device=state.next_logits.device,
    )
    expected_next = model.select_state(model.advance_state(state, token_ids), indices)
    selected_tokens = token_ids.index_select(0, indices)
    actual_selected = model.advance_state(selected, selected_tokens)
    actual_stacked = model.advance_state(stacked, selected_tokens)
    torch.testing.assert_close(actual_selected.next_logits, expected_next.next_logits)
    torch.testing.assert_close(actual_stacked.next_logits, expected_next.next_logits)


def test_empty_batch_scores_are_empty(tokenizer):
    model = AutoregressiveGRU(tokenizer, embedding_dim=4, hidden_size=5)
    assert model.log_probabilities([]).shape == (0,)
    assert model.surprisal_bits_batch([]).shape == (0,)


def test_password_model_remains_abstract():
    assert inspect.isabstract(PasswordModel)
    with pytest.raises(TypeError):
        PasswordModel()
