import pytest

from scripts.monitoring import TensorBoardMonitor


class FakeWriter:
    def __init__(self):
        self.scalars = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, scalar_value, global_step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


def test_monitor_groups_scalars_and_adapts_progress_callback():
    writer = FakeWriter()
    monitor = TensorBoardMonitor(writer)

    monitor.log_metrics("Loss", {"train": 1.2, "validation": 1.3}, 2, flush=True)
    callback = monitor.progress_callback("Surprisal/GRU")
    callback(100, {"progress": 0.5})
    monitor.close()

    assert writer.scalars == [
        ("Loss/train", 1.2, 2),
        ("Loss/validation", 1.3, 2),
        ("Surprisal/GRU/progress", 0.5, 100),
    ]
    assert writer.flush_count == 1
    assert writer.closed


@pytest.mark.parametrize(
    ("group", "metrics", "step"),
    [("", {"value": 1.0}, 0), ("Loss", {"": 1.0}, 0), ("Loss", {"x": float("nan")}, 0), ("Loss", {"x": 1.0}, -1)],
)
def test_monitor_rejects_invalid_scalar_records(group, metrics, step):
    with pytest.raises(ValueError):
        TensorBoardMonitor(FakeWriter()).log_metrics(group, metrics, step)
