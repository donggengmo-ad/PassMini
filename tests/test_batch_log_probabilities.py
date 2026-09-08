import math

import pytest
import torch

from scripts.models import AutoregressiveTCN, AutoregressiveTransformer
from scripts.tokenizer import CharTokenizer


@pytest.fixture(params=["tcn", "transformer"])
def model(request):
    tokenizer = CharTokenizer.from_text(["abc"])
    if request.param == "tcn":
        instance = AutoregressiveTCN(
            tokenizer,
            embedding_dim=4,
            channels=6,
            dilations=(1, 2),
        )
    else:
        instance = AutoregressiveTransformer(
            tokenizer,
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=12,
            max_length=8,
        )
    instance.eval()
    return instance


def _manual_log_probability(model, text):
    """独立构造单条 input/target，作为批量实现的数值参照。"""

    ids = model.tokenizer.encode(text)
    input_ids = torch.tensor(
        [[model.tokenizer.bos_id, *ids]], dtype=torch.long, device=model.device
    )
    target_ids = torch.tensor(
        [[*ids, model.tokenizer.eos_id]], dtype=torch.long, device=model.device
    )
    token_log_probs = torch.log_softmax(model(input_ids), dim=-1)
    return token_log_probs.gather(-1, target_ids.unsqueeze(-1)).sum()


def test_batch_log_probabilities_match_independent_single_sequence_scores(model):
    texts = ["abc", "", "a", "bc"]

    expected = torch.stack([_manual_log_probability(model, text) for text in texts])
    actual = model.log_probabilities(texts)

    torch.testing.assert_close(actual, expected)


def test_batch_log_probabilities_use_one_forward_and_include_eos(model):
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    forward_calls = []
    handle = model.register_forward_hook(
        lambda _module, _inputs, _output: forward_calls.append(1)
    )

    texts = ["", "a", "abc"]
    actual = model.log_probabilities(texts)
    handle.remove()
    expected = torch.tensor(
        [-(len(text) + 1) * math.log(model.vocab_size) for text in texts]
    )

    assert len(forward_calls) == 1
    torch.testing.assert_close(actual.cpu(), expected)


def test_batch_log_probabilities_handle_empty_batch_on_model_device(model):
    scores = model.log_probabilities([])

    assert scores.shape == (0,)
    assert scores.device == model.device
