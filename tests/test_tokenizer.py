import pytest
from scripts import tokenizer
from pathlib import Path

def test_char_tokenizer():
    token = tokenizer.CharTokenizer.from_text(["hello world"])
    assert token.decode(token.encode("hello world")) == "hello world"
    assert token.decode(token.encode("a"), skip_special_tokens=False) == "UNK"
    assert token.decode(
        token.encode("hello world",
                     add_bos=True,
                     add_eos=True
        ),
        skip_special_tokens=False,
        stop_at_eos=False
    ) == "BOShello worldEOS"
    for char in token.id_to_token:
        assert token.id_to_token[token.token_to_id[char]] == char

def test_char_tokenizer_json(tmp_path: Path):
    token = tokenizer.CharTokenizer.from_text(["hello world"])
    json_path = tmp_path / "test.json"
    token.dump(json_path)
    loaded_token = tokenizer.CharTokenizer.from_json(json_path)
    assert token.encode("hello world") == loaded_token.encode("hello world")
    assert token == loaded_token
