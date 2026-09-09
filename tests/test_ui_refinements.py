"""R13–R15 与 R18 的行为回归。"""

from unittest.mock import patch

import pytest

from scripts.experiment import (
    AutoregressiveBigramConfig, AutoregressiveMLPConfig, AutoregressiveGRUConfig,
    AutoregressiveTCNConfig, AutoregressiveTransformerConfig,
)
from scripts.model_factory import build_model_from_config
from scripts.pipeline import build_model
from scripts.tokenizer import CharTokenizer
from app.frontend.playground import trace_character_probabilities, trace_temperature_probabilities
from app.frontend.playground_session import remember_result, recall_result, clear_playground_results


@pytest.mark.parametrize("config", [
    AutoregressiveBigramConfig(),
    AutoregressiveMLPConfig(tau=2, embedding_dim=4, hidden_size=8),
    AutoregressiveGRUConfig(embedding_dim=4, hidden_size=8),
    AutoregressiveTCNConfig(embedding_dim=4, channels=8, dilations=(1, 2)),
    AutoregressiveTransformerConfig(d_model=4, nhead=2, num_layers=1, dim_feedforward=8),
])
def test_temperatures_share_one_forward_and_preserve_distributions(config):
    tokenizer = CharTokenizer.from_text(["ab12"])
    model = build_model_from_config(config, tokenizer).eval()
    runtime = {"model": (model, tokenizer)}
    temperatures = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    with patch.object(model, "forward", wraps=model.forward) as forward:
        actual = trace_temperature_probabilities("ab1", runtime, temperatures)
    assert forward.call_count == 1
    for temperature in temperatures:
        expected = trace_character_probabilities("ab1", runtime, temperature=temperature)[0]
        assert actual[temperature] == expected
    with patch("scripts.pipeline.build_model_from_config", return_value=model) as factory:
        assert build_model(config, tokenizer) is model
        factory.assert_called_once_with(config, tokenizer)


def test_training_and_inference_import_the_same_factory():
    from scripts import inference, pipeline
    assert inference.build_model_from_config is build_model_from_config
    assert pipeline.build_model_from_config is build_model_from_config


def test_session_results_are_bounded_isolated_and_selection_aware():
    session, other = {}, {}
    remember_result(session, "manual", ["a"], "first")
    remember_result(session, "manual", ["a"], "latest")
    assert recall_result(session, "manual", ["a"]) == "latest"
    assert recall_result(other, "manual", ["a"]) is None
    assert len(session["playground_results"]) == 1
    assert recall_result(session, "manual", ["b"]) is None
    assert recall_result(session, "manual", ["a"]) is None
    remember_result(session, "beam", ["a"], "demo")
    session["character_lab_input_text"] = "abc"
    clear_playground_results(session)
    assert session == {}


@pytest.mark.parametrize("temperatures", [(), (0,), (float("nan"),), (float("inf"),)])
def test_invalid_temperatures_fail_before_forward(temperatures):
    tokenizer = CharTokenizer.from_text(["ab"])
    model = build_model_from_config(AutoregressiveBigramConfig(), tokenizer)
    with patch.object(model, "forward") as forward:
        with pytest.raises(ValueError):
            trace_temperature_probabilities("ab", {"a": (model, tokenizer)}, temperatures)
    forward.assert_not_called()
