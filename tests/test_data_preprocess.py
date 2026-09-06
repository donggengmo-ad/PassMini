import hashlib
import random
from pathlib import Path

import pytest

from scripts.data import (normalize_password, prepare_dataset)


def read_passwords(path: Path) -> list[str]:
    """Read the one-password-per-line format produced by preprocessing."""
    return path.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize(
    ("raw_line", "expected"),
    [
        ("a\n", "a"),
        ("twelve_chars\r\n", "twelve_chars"),
        (" leading ", " leading "),
        ("~", "~"),
    ],
)
def test_normalize_password_accepts_valid_ascii_and_only_removes_line_endings(
    raw_line: str,
    expected: str,
):
    assert normalize_password(raw_line) == expected


@pytest.mark.parametrize(
    "raw_line",
    [
        "\n",
        "thirteen_char\n",
        "has\ttab\n",
        "pass\x7fword\n",
        "密码\n",
    ],
)
def test_normalize_password_rejects_invalid_length_or_charset(raw_line: str):
    assert normalize_password(raw_line) is None


def test_normalize_password_honors_custom_length_bounds():
    assert normalize_password("abc\n", min_length=3, max_length=3) == "abc"
    assert normalize_password("ab\n", min_length=3, max_length=3) is None
    assert normalize_password("abcd\n", min_length=3, max_length=3) is None


def write_source(path: Path, passwords: list[str]) -> None:
    path.write_text("\n".join(passwords) + "\n", encoding="utf-8")


def test_prepare_dataset_is_reproducible_disjoint_and_deduplicated(tmp_path: Path):
    source_path = tmp_path / "source.txt"
    passwords = [f"sample{i:02d}" for i in range(20)]
    write_source(source_path, passwords + ["sample03", "sample11"])

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_stats = prepare_dataset(source_path, first_dir, 8, 3, 4, seed=42)
    second_stats = prepare_dataset(source_path, second_dir, 8, 3, 4, seed=42)

    assert first_stats == second_stats
    for filename in ("train.txt", "val.txt", "test.txt"):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()

    train = read_passwords(first_dir / "train.txt")
    val = read_passwords(first_dir / "val.txt")
    test = read_passwords(first_dir / "test.txt")

    assert (len(train), len(val), len(test)) == (8, 3, 4)
    assert len(set(train)) == len(train)
    assert len(set(val)) == len(val)
    assert len(set(test)) == len(test)
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)


def test_prepare_dataset_filters_invalid_and_duplicate_passwords(tmp_path: Path):
    source_path = tmp_path / "source.txt"
    source_path.write_text(
        "valid1\n"
        "valid2\n"
        "valid2\n"
        "\n"
        "thirteen_char\n"
        "has\ttab\n"
        "密码\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    prepare_dataset(source_path, output_dir, 1, 0, 1, seed=7)

    actual = read_passwords(output_dir / "train.txt")
    actual += read_passwords(output_dir / "val.txt")
    actual += read_passwords(output_dir / "test.txt")

    assert sorted(actual) == ["valid1", "valid2"]


def test_prepare_dataset_rejects_lines_with_invalid_utf8_bytes(tmp_path: Path):
    source_path = tmp_path / "source.txt"
    source_path.write_bytes(b"valid1\ninvalid\xf1line\nvalid2\n")

    output_dir = tmp_path / "output"
    prepare_dataset(source_path, output_dir, 1, 0, 1, seed=7)

    actual = read_passwords(output_dir / "train.txt")
    actual += read_passwords(output_dir / "test.txt")

    assert sorted(actual) == ["valid1", "valid2"]


def test_prepare_dataset_preserves_ascii_password_text_exactly(tmp_path: Path):
    source_path = tmp_path / "source.txt"
    passwords = ["comma,value", 'double"quote', " leading", "trailing "]
    write_source(source_path, passwords)

    output_dir = tmp_path / "output"
    prepare_dataset(source_path, output_dir, 2, 1, 1, seed=3)

    actual = []
    for filename in ("train.txt", "val.txt", "test.txt"):
        actual.extend(read_passwords(output_dir / filename))

    assert sorted(actual) == sorted(passwords)


def test_prepare_dataset_raises_when_unique_valid_input_is_insufficient(tmp_path: Path):
    source_path = tmp_path / "source.txt"
    write_source(source_path, ["only1", "only2", "only2", "密码"])

    with pytest.raises(ValueError, match="至少|不同密码|unique|有效|不足"):
        prepare_dataset(source_path, tmp_path / "output", 2, 1, 0, seed=1)


def test_prepare_dataset_selects_passwords_with_smallest_hashes(tmp_path: Path):
    source_path = tmp_path / "source.txt"
    passwords = [f"pw{i:02d}" for i in range(30)]
    write_source(source_path, passwords + ["pw03", "pw11", "pw29"])
    seed = 42
    sample_size = 15

    output_dir = tmp_path / "output"
    prepare_dataset(source_path, output_dir, 8, 3, 4, seed=seed)

    expected = set(
        sorted(
            set(passwords),
            key=lambda password: hashlib.blake2b(
                f"{seed}\0{password}".encode("ascii"),
                digest_size=16,
            ).digest(),
        )[:sample_size]
    )
    actual = set()
    for filename in ("train.txt", "val.txt", "test.txt"):
        actual.update(read_passwords(output_dir / filename))

    assert actual == expected


def test_prepare_dataset_is_independent_of_source_order(tmp_path: Path):
    passwords = [f"pw{i:02d}" for i in range(30)] + ["pw03", "pw11", "pw29"]
    forward_path = tmp_path / "forward.txt"
    reversed_path = tmp_path / "reversed.txt"
    write_source(forward_path, passwords)
    write_source(reversed_path, list(reversed(passwords)))

    forward_dir = tmp_path / "forward"
    reversed_dir = tmp_path / "reversed"
    prepare_dataset(forward_path, forward_dir, 8, 3, 4, seed=42)
    prepare_dataset(reversed_path, reversed_dir, 8, 3, 4, seed=42)

    for filename in ("train.txt", "val.txt", "test.txt"):
        assert (forward_dir / filename).read_bytes() == (
            reversed_dir / filename
        ).read_bytes()


def test_prepare_dataset_does_not_change_global_random_state(tmp_path: Path):
    source_path = tmp_path / "source.txt"
    write_source(source_path, [f"pw{i:02d}" for i in range(20)])

    original_state = random.getstate()
    try:
        random.seed(9876)
        state_before = random.getstate()
        prepare_dataset(source_path, tmp_path / "output", 8, 3, 4, seed=42)
        state_after = random.getstate()
    finally:
        random.setstate(original_state)

    assert state_after == state_before
