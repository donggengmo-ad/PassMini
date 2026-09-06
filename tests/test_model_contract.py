import inspect

import pytest
import torch

from scripts.models import (
    AutoregressiveBigram,
    AutoregressiveGRU,
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


def test_empty_batch_scores_are_empty(tokenizer):
    model = AutoregressiveGRU(tokenizer, embedding_dim=4, hidden_size=5)
    assert model.log_probabilities([]).shape == (0,)
    assert model.surprisal_bits_batch([]).shape == (0,)


def test_password_model_remains_abstract():
    assert inspect.isabstract(PasswordModel)
    with pytest.raises(TypeError):
        PasswordModel()
