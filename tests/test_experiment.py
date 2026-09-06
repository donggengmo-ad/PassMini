import json
from pathlib import Path

import pytest
import torch

from scripts.experiment import (
    AutoregressiveBigramConfig,
    AutoregressiveGRUConfig,
    AutoregressiveTCNConfig,
    AutoregressiveTransformerConfig,
    ExperimentConfig,
    SchedulerConfig,
    model_config_from_dict,
)


def test_experiment_defaults_are_canonical_and_independent():
    first = ExperimentConfig()
    second = ExperimentConfig()

    assert isinstance(first.model, AutoregressiveGRUConfig)
    assert first.model.model_type == "gru"
    assert first.data.output_dir == Path("data/experiments/autorg_gru")
    assert first.training.scheduler.name == "none"
    assert first.training is not second.training
    assert first.training.scheduler is not second.training.scheduler
    first.training.batch_size = 32
    assert second.training.batch_size == 256


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"model_type": "bigram", "alpha": 0.5}, AutoregressiveBigramConfig),
        ({"model_type": "gru", "embedding_dim": 8}, AutoregressiveGRUConfig),
        ({"model_type": "tcn", "channels": 8}, AutoregressiveTCNConfig),
        ({"model_type": "transformer", "d_model": 8, "nhead": 2}, AutoregressiveTransformerConfig),
    ],
)
def test_model_config_from_dict_uses_canonical_tags(payload, expected):
    config = model_config_from_dict(payload)
    assert isinstance(config, expected)
    assert config.model_type == payload["model_type"]


@pytest.mark.parametrize("model_type", ["causal_cnn", "mini_transformer", "", None, "legacy"])
def test_model_config_rejects_legacy_or_missing_tags(model_type):
    with pytest.raises(ValueError):
        model_config_from_dict({"model_type": model_type})


@pytest.mark.parametrize("payload", [{"model_type": "gru", "mlp_hidden_size": 32}, {"model_type": "gru", "weight_tying": False}])
def test_model_config_rejects_legacy_fields(payload):
    with pytest.raises(ValueError, match="不支持的字段"):
        model_config_from_dict(payload)


def test_model_configs_validate_structure():
    with pytest.raises(ValueError):
        AutoregressiveGRUConfig(hidden_size=0)
    with pytest.raises(ValueError):
        AutoregressiveTCNConfig(dilations=())
    with pytest.raises(ValueError):
        AutoregressiveTransformerConfig(d_model=5, nhead=2)


def test_config_to_dict_is_json_serializable():
    config = ExperimentConfig()
    config.data.train_limit = 100
    value = config.to_dict()
    assert value["model"]["model_type"] == "gru"
    assert value["data"]["dataset_path"] == "data/processed"
    assert json.loads(json.dumps(value)) == value


@pytest.mark.parametrize("name", ["none", "reduce_on_plateau", "step_lr", "cosine"])
def test_scheduler_config_accepts_supported_names(name):
    config = SchedulerConfig(name=name)
    assert config.name == name


def test_scheduler_config_validates_parameters():
    with pytest.raises(ValueError):
        SchedulerConfig(name="step_lr", step_size=0)
    with pytest.raises(ValueError):
        SchedulerConfig(name="step_lr", gamma=0)
    with pytest.raises(ValueError):
        SchedulerConfig(name="cosine", t_max=0)
    with pytest.raises(ValueError):
        SchedulerConfig(name="unknown")


def test_smoke_cpu_resolves_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = ExperimentConfig()
    config.device.mode = "smoke_cpu"
    assert config.resolve_device() == torch.device("cpu")


def test_formal_cuda_requires_available_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = ExperimentConfig()
    with pytest.raises(RuntimeError, match="CUDA"):
        config.resolve_device()


def test_formal_cuda_resolves_valid_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    config = ExperimentConfig()
    config.device.cuda_index = 1
    assert config.resolve_device() == torch.device("cuda:1")


def test_formal_cuda_rejects_invalid_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    config = ExperimentConfig()
    config.device.cuda_index = 2
    with pytest.raises(ValueError, match="cuda_index"):
        config.resolve_device()
