import json

from scripts.experiment import (
    AutoregressiveGRUConfig,
    DataConfig,
    DeviceConfig,
    ExperimentConfig,
    SchedulerConfig,
    TrainingConfig,
)
from scripts.pipeline import build_model, run_training_experiment
from scripts.tokenizer import CharTokenizer


def make_config(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    splits = {"train": ["ab", "abc", "a"], "val": ["ab"], "test": ["abc"]}
    for name, values in splits.items():
        (data_dir / f"{name}.txt").write_text("\n".join(values), encoding="utf-8")
    tokenizer = CharTokenizer.from_text(splits["train"])
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer.dump(tokenizer_path)
    return ExperimentConfig(
        data=DataConfig(dataset_path=data_dir, tokenizer_path=tokenizer_path, output_dir=tmp_path / "output"),
        model=AutoregressiveGRUConfig(embedding_dim=4, hidden_size=5),
        training=TrainingConfig(batch_size=2, num_epochs=1, scheduler=SchedulerConfig(name="step_lr")),
        device=DeviceConfig(mode="smoke_cpu"),
    )


def test_pipeline_trains_and_saves_canonical_artifacts(tmp_path):
    config = make_config(tmp_path)
    artifacts = run_training_experiment(config)
    assert artifacts.model.model_type == "gru"
    assert artifacts.test_loss is not None
    assert (config.data.output_dir / "checkpoint_latest.pt").is_file()
    assert (config.data.output_dir / "model.pt").is_file()
    assert (config.data.output_dir / "inference.json").is_file()
    saved_config = json.loads((config.data.output_dir / "inference.json").read_text())
    assert saved_config["model_type"] == "gru"


def test_build_model_accepts_model_config_directly(tmp_path):
    tokenizer = CharTokenizer.from_text(["ab"])
    model = build_model(AutoregressiveGRUConfig(embedding_dim=4, hidden_size=5), tokenizer)
    assert model.model_type == "gru"
