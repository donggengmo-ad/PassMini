import json

from scripts.experiment import (
    AutoregressiveGRUConfig,
    AutoregressiveMLPConfig,
    DataConfig,
    DeviceConfig,
    ExperimentConfig,
    SchedulerConfig,
    TrainingConfig,
)
from scripts.pipeline import build_model, run_training_experiment
from scripts.monitoring import TensorBoardMonitor
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


def test_build_model_accepts_mlp_config():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = build_model(
        AutoregressiveMLPConfig(tau=4, embedding_dim=4, hidden_size=6),
        tokenizer,
    )

    assert model.model_type == "mlp"
    assert model.tau == 4


def test_mlp_pipeline_trains_and_saves_canonical_artifacts(tmp_path):
    config = make_config(tmp_path)
    config.model = AutoregressiveMLPConfig(
        tau=2,
        embedding_dim=4,
        hidden_size=6,
    )

    artifacts = run_training_experiment(config)

    assert artifacts.model.model_type == "mlp"
    assert artifacts.test_loss is not None
    inference = json.loads(
        (config.data.output_dir / "inference.json").read_text(encoding="utf-8")
    )
    assert inference == {
        "model_type": "mlp",
        "tau": 2,
        "embedding_dim": 4,
        "hidden_size": 6,
    }


def test_pipeline_creates_and_closes_optional_tensorboard_monitor(tmp_path, monkeypatch):
    class FakeWriter:
        def __init__(self):
            self.scalars = []
            self.closed = False

        def add_scalar(self, tag, scalar_value, global_step):
            self.scalars.append((tag, float(scalar_value), global_step))

        def flush(self):
            pass

        def close(self):
            self.closed = True

    writer = FakeWriter()
    monitor = TensorBoardMonitor(writer)
    monkeypatch.setattr(
        TensorBoardMonitor,
        "create",
        classmethod(lambda cls, log_dir: monitor),
    )

    config = make_config(tmp_path)
    run_training_experiment(
        config,
        tensorboard_log_dir=tmp_path / "tensorboard",
        tensorboard_log_interval=1,
    )

    assert writer.closed
    assert any(tag == "Loss/test" for tag, _, _ in writer.scalars)
