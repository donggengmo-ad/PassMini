import torch
import pytest
from pathlib import Path

from scripts import data, tokenizer
from scripts.data import PasswordDataset, collate_batch
from scripts.data import read_dataset


def test_dataset():
    token = tokenizer.CharTokenizer.from_text(["hello world"])
    dataset = data.PasswordDataset(["hello", "world"], token)
    assert token == dataset.tokenizer
    assert len(dataset) == 2
    input_tensor, target_tensor = dataset[0] # hello
    assert input_tensor.dtype == torch.long
    assert target_tensor.dtype == torch.long
    assert input_tensor.shape == torch.Size([6])
    assert target_tensor.shape == torch.Size([6])
    torch.testing.assert_close(input_tensor[1:], target_tensor[:-1])
    assert input_tensor[0].item() == token.bos_id
    assert target_tensor[-1].item() == token.eos_id

def test_collate_batch():
    token = tokenizer.CharTokenizer.from_text(["I am kind of sleepy now"])
    dataset = PasswordDataset(["I", "am", "kind", "of", "sleepy", "now"], token)
    batch = [dataset[i] for i in range(len(dataset))]
    input_batch, target_batch = collate_batch(batch, token.pad_id)
    assert input_batch.dtype == torch.long
    assert target_batch.dtype == torch.long
    assert input_batch.shape == torch.Size([6, 7])
    assert target_batch.shape == torch.Size([6, 7])
    for i, (input_tensor, target_tensor) in enumerate(batch):
        assert torch.all(input_batch[i, len(input_tensor):] == token.pad_id)
        assert torch.all(target_batch[i, len(target_tensor):] == token.pad_id)


def test_read_dataset_loads_all_splits_and_applies_individual_limits(tmp_path: Path):
    (tmp_path / "train.txt").write_text("train one\ntrain two\ntrain three\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("val one\nval two\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("test one\ntest two\ntest three\n", encoding="utf-8")

    dataset = read_dataset(tmp_path, train_limit=2, val_limit=1, test_limit=None)

    assert dataset == {
        "train": ["train one", "train two"],
        "val": ["val one"],
        "test": ["test one", "test two", "test three"],
    }


def test_read_dataset_removes_only_line_endings(tmp_path: Path):
    (tmp_path / "train.txt").write_text(" leading \ntrailing  \n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("value\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("value\n", encoding="utf-8")

    dataset = read_dataset(tmp_path)

    assert dataset["train"] == [" leading ", "trailing  "]


def test_read_dataset_requires_each_split_file(tmp_path: Path):
    (tmp_path / "train.txt").write_text("train\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("val\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="test.txt"):
        read_dataset(tmp_path)
