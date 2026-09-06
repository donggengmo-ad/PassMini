import matplotlib
import pytest

matplotlib.use("Agg")

from scripts.util import plot_loss


def test_plot_loss_marks_test_loss_with_horizontal_dashed_line():
    import matplotlib.pyplot as plt

    figure = plt.figure()
    history = {
        "train_loss": [3.0, 2.0, 1.5],
        "valid_loss": [2.5, 1.8, 2.0],
    }

    plot_loss(history, test_loss=1.9)

    axis = plt.gca()
    assert [list(line.get_xdata()) for line in axis.lines] == [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1],
    ]
    test_line = axis.lines[-1]
    assert list(test_line.get_ydata()) == [1.9, 1.9]
    assert test_line.get_linestyle() == "--"
    assert test_line.get_label() == "Test Loss"
    assert axis.get_xlabel() == "Epochs"
    assert axis.get_ylabel() == "Loss"

    plt.close(figure)


def test_plot_loss_does_not_draw_test_line_by_default():
    import matplotlib.pyplot as plt

    figure = plt.figure()

    plot_loss({"train_loss": [3.0], "valid_loss": [2.0]})

    assert len(plt.gca().lines) == 2
    plt.close(figure)


def test_plot_loss_keeps_save_path_behavior(tmp_path):
    import matplotlib.pyplot as plt

    figure = plt.figure()
    path = tmp_path / "loss.png"

    plot_loss({"train_loss": [1.0], "valid_loss": [1.0]}, save_path=path)

    assert path.exists()
    plt.close(figure)
