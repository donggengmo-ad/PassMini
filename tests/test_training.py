from functools import partial
from math import isfinite

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.data import PasswordDataset, collate_batch
from scripts.experiment import SchedulerConfig
from scripts.models import AutoregressiveGRU
from scripts.monitoring import TensorBoardMonitor
from scripts.tokenizer import CharTokenizer
from scripts.training import (
    build_scheduler,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train,
    train_one_epoch,
)


WORDS = ["password", "123456", "qwerty", "abc123", "letmein", "monkey", "dragon"]
TOKENIZER = CharTokenizer.from_text(WORDS)
DATASET = PasswordDataset(WORDS, TOKENIZER)
COLLATE = partial(collate_batch, pad_id=TOKENIZER.pad_id)


def make_training_objects():
    loader = DataLoader(DATASET, batch_size=3, collate_fn=COLLATE, shuffle=False)
    model = AutoregressiveGRU(TOKENIZER, embedding_dim=8, hidden_size=12)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=TOKENIZER.pad_id, reduction="sum")
    return model, loader, criterion


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_train_one_epoch_updates_parameters_and_returns_finite_loss():
    model, loader, criterion = make_training_objects()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    before = snapshot(model)
    loss = train_one_epoch(model, loader, optimizer, criterion=criterion)
    assert isfinite(loss)
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert model.training


def test_evaluate_reads_model_pad_id_and_does_not_update_parameters():
    model, loader, criterion = make_training_objects()
    before = snapshot(model)
    loss = evaluate(model, loader, criterion=criterion)
    torch.testing.assert_close(before, model.state_dict())
    assert isfinite(loss)
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluate_uses_non_pad_token_average():
    tokenizer = CharTokenizer(["PAD", "BOS", "EOS", "UNK", "a"])

    class FixedModel(torch.nn.Module):
        model_type = "test"

        def __init__(self):
            super().__init__()
            self.tokenizer = tokenizer
            self.logits = torch.nn.Parameter(torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]]))

        @property
        def pad_id(self):
            return self.tokenizer.pad_id

        def forward(self, inputs):
            return self.logits.expand(inputs.size(0), inputs.size(1), -1)

    inputs = torch.zeros((2, 3), dtype=torch.long)
    targets = torch.tensor([[1, 2, 1], [1, tokenizer.pad_id, tokenizer.pad_id]])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction="sum")
    actual = evaluate(FixedModel(), loader, criterion=criterion)
    expected = (3 * torch.nn.functional.cross_entropy(torch.tensor([[2.0, 0, 0, 0, 0]]), torch.tensor([1]))
                + torch.nn.functional.cross_entropy(torch.tensor([[2.0, 0, 0, 0, 0]]), torch.tensor([2]))).item() / 4
    assert actual == pytest.approx(expected)


def test_criterion_validation_uses_model_pad_id():
    model, loader, _ = make_training_objects()
    wrong_reduction = torch.nn.CrossEntropyLoss(ignore_index=model.pad_id, reduction="mean")
    with pytest.raises(ValueError):
        train_one_epoch(model, loader, torch.optim.Adam(model.parameters()), criterion=wrong_reduction)
    wrong_pad = torch.nn.CrossEntropyLoss(ignore_index=model.pad_id + 1, reduction="sum")
    with pytest.raises(ValueError):
        evaluate(model, loader, criterion=wrong_pad)


@pytest.mark.parametrize("name", ["none", "reduce_on_plateau", "step_lr", "cosine"])
def test_build_scheduler_supports_all_strategies(name):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_scheduler(optimizer, SchedulerConfig(name=name, step_size=2, gamma=0.5), 5)
    if name == "none":
        assert scheduler is None
    else:
        assert scheduler is not None


def test_step_lr_changes_learning_rate_at_step_size():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_scheduler(optimizer, SchedulerConfig(name="step_lr", step_size=2, gamma=0.5), 5)
    assert scheduler is not None
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_checkpoint_round_trip_includes_scheduler_and_model_config(tmp_path):
    model, loader, criterion = make_training_objects()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler_config = SchedulerConfig(name="step_lr", step_size=2, gamma=0.5)
    scheduler = build_scheduler(optimizer, scheduler_config, 3)
    train_one_epoch(model, loader, optimizer, criterion=criterion)
    scheduler.step()
    path = tmp_path / "nested" / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        model_config={"model_type": "gru", "embedding_dim": 8, "hidden_size": 12, "num_layers": 1},
        scheduler=scheduler,
        scheduler_config=scheduler_config,
        history={"learning_rate": [0.01]},
        best_val_loss=1.2,
    )
    checkpoint = load_checkpoint(path)
    assert checkpoint["model_config"]["model_type"] == "gru"
    assert checkpoint["scheduler_state_dict"] is not None
    assert checkpoint["scheduler_config"]["name"] == "step_lr"
    assert checkpoint["best_val_loss"] == pytest.approx(1.2)
    assert checkpoint["history"]["learning_rate"] == [0.01]


def test_train_records_learning_rate_and_saves_latest_and_best(tmp_path):
    model, loader, criterion = make_training_objects()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = train(
        model,
        loader,
        loader,
        optimizer,
        criterion=criterion,
        num_epochs=2,
        save_path=tmp_path,
        model_config={"model_type": "gru", "embedding_dim": 8, "hidden_size": 12, "num_layers": 1},
    )
    assert len(history["learning_rate"]) == 2
    assert (tmp_path / "checkpoint_latest.pt").is_file()
    assert (tmp_path / "checkpoint_best.pt").is_file()


def test_train_logs_batch_and_epoch_metrics_to_monitor(tmp_path):
    class FakeWriter:
        def __init__(self):
            self.scalars = []

        def add_scalar(self, tag, scalar_value, global_step):
            self.scalars.append((tag, float(scalar_value), global_step))

        def flush(self):
            pass

        def close(self):
            pass

    model, loader, criterion = make_training_objects()
    writer = FakeWriter()
    monitor = TensorBoardMonitor(writer)

    train(
        model,
        loader,
        loader,
        torch.optim.Adam(model.parameters(), lr=0.01),
        criterion=criterion,
        num_epochs=1,
        save_path=tmp_path,
        monitor=monitor,
        monitor_log_interval=1,
    )

    tags = {tag for tag, _, _ in writer.scalars}
    assert "Train/Batch/running_loss" in tags
    assert "Validation/Batch/running_loss" in tags
    assert {"Loss/train", "Loss/validation", "Loss/generalization_gap"} <= tags
    assert {"Optimization/learning_rate", "Runtime/epoch_seconds"} <= tags
