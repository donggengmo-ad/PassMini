import math

import pytest
import torch

from scripts.models import AutoregressiveGRU
from scripts.tokenizer import CharTokenizer


def make_zero_model(tokenizer):
    model = AutoregressiveGRU(tokenizer, embedding_dim=4, hidden_size=5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


class ScriptedGenerationModel(AutoregressiveGRU):
    def __init__(self, tokenizer, step_scores):
        super().__init__(tokenizer, embedding_dim=2, hidden_size=3)
        self.step_scores = step_scores
        self.step = 0

    def forward(self, input_ids, hidden=None, return_hidden=False):
        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), self.vocab_size),
            -torch.inf,
            device=input_ids.device,
        )
        scores = self.step_scores[min(self.step, len(self.step_scores) - 1)]
        for token_id, score in scores.items():
            logits[..., token_id] = score
        self.step += 1
        if hidden is None:
            hidden = torch.zeros(self.num_layers, input_ids.size(0), self.hidden_size, device=input_ids.device)
        return (logits, hidden) if return_hidden else logits


def test_weight_tying_is_fixed_and_old_fields_are_absent():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveGRU(tokenizer, embedding_dim=4, hidden_size=5)
    assert not hasattr(model, "mlp")
    assert not hasattr(model, "weight_tying")
    assert model.output_projection.out_features == model.embedding_dim
    assert model(torch.tensor([[tokenizer.bos_id]])).shape[-1] == tokenizer.vocab_size


def test_log_probabilities_include_eos_and_ignore_padding():
    tokenizer = CharTokenizer.from_text(["a", "abc"])
    model = make_zero_model(tokenizer)
    scores = model.log_probabilities(["", "a", "abc"])
    expected = torch.tensor([-math.log(tokenizer.vocab_size), -2 * math.log(tokenizer.vocab_size), -4 * math.log(tokenizer.vocab_size)])
    torch.testing.assert_close(scores, expected)


def test_surprisal_bits_is_log_probability_conversion():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = make_zero_model(tokenizer)
    scores = model.surprisal_bits_batch(["ab"])
    expected = -model.log_probabilities(["ab"]) / math.log(2)
    torch.testing.assert_close(scores, expected)


def test_generate_stops_at_eos_and_filters_special_tokens():
    tokenizer = CharTokenizer.from_text(["a"])
    a_id = tokenizer.token_to_id["a"]
    model = ScriptedGenerationModel(
        tokenizer,
        [{tokenizer.pad_id: 100, tokenizer.bos_id: 100, tokenizer.unk_id: 100, a_id: 0}, {tokenizer.eos_id: 0}],
    )
    assert model.generate(max_length=5) == "a"


def test_generate_stops_at_max_length_without_eos():
    tokenizer = CharTokenizer.from_text(["a"])
    model = ScriptedGenerationModel(tokenizer, [{tokenizer.token_to_id["a"]: 0}])
    assert model.generate(max_length=4) == "aaaa"


def test_generate_batch_keeps_fixed_batch_and_reproducibility():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = make_zero_model(tokenizer)
    first = model.generate_batch(5, 6, generator=torch.Generator().manual_seed(42))
    second = model.generate_batch(5, 6, generator=torch.Generator().manual_seed(42))
    assert first == second
    assert len(first) == 5
    assert all(len(value) <= 6 for value in first)


def test_generate_batch_finished_rows_remain_in_batch():
    tokenizer = CharTokenizer.from_text(["a"])

    class RowModel(ScriptedGenerationModel):
        def __init__(self, tokenizer):
            super().__init__(tokenizer, [{tokenizer.eos_id: 0, tokenizer.token_to_id["a"]: -10}])
            self.batch_sizes = []

        def forward(self, input_ids, hidden=None, return_hidden=False):
            self.batch_sizes.append(input_ids.size(0))
            return super().forward(input_ids, hidden, return_hidden)

    model = RowModel(tokenizer)
    assert len(model.generate_batch(4, 3)) == 4
    assert model.batch_sizes and all(size == 4 for size in model.batch_sizes)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": -1}, {"max_length": 0}, {"temperature": 0.0}])
def test_generate_batch_rejects_invalid_arguments(kwargs):
    model = make_zero_model(CharTokenizer.from_text(["a"]))
    with pytest.raises(ValueError):
        model.generate_batch(**{"batch_size": 2, "max_length": 4, **kwargs})
