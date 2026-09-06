import json
from dataclasses import asdict

import pytest
import torch

from scripts.experiment import (
    AutoregressiveGRUConfig,
    AutoregressiveTCNConfig,
    AutoregressiveTransformerConfig,
)
from scripts.inference import (
    GenerationCandidate,
    beam_search,
    best_first_search,
    depth_first_search,
    inference_config_dict,
    load_generation_candidates,
    load_inference_model,
    save_generation_candidates,
    save_inference_config,
    score_password,
    score_passwords,
)
from scripts.models import (
    AutoregressiveBigram,
    AutoregressiveGRU,
    AutoregressiveTCN,
    AutoregressiveTransformer,
)
from scripts.tokenizer import CharTokenizer


def make_gru_artifact(tmp_path):
    tokenizer = CharTokenizer.from_text(["ab", "abc"])
    model = AutoregressiveGRU(tokenizer, embedding_dim=4, hidden_size=5, num_layers=2)
    model_path = tmp_path / "model.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    config_path = tmp_path / "inference.json"
    torch.save(model.state_dict(), model_path)
    tokenizer.dump(tokenizer_path)
    config = AutoregressiveGRUConfig(embedding_dim=4, hidden_size=5, num_layers=2)
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    return model, tokenizer, model_path, tokenizer_path, config_path


def test_load_inference_model_restores_gru(tmp_path):
    model, tokenizer, model_path, tokenizer_path, config_path = make_gru_artifact(tmp_path)
    loaded, loaded_tokenizer = load_inference_model(model_path, tokenizer_path, config_path)
    assert loaded_tokenizer == tokenizer
    assert not loaded.training
    torch.testing.assert_close(loaded.state_dict(), model.state_dict())


@pytest.mark.parametrize(
    ("model", "config"),
    [
        (
            AutoregressiveTCN(
                CharTokenizer.from_text(["abc"]),
                embedding_dim=4,
                channels=6,
                dilations=(1, 2),
            ),
            AutoregressiveTCNConfig(
                embedding_dim=4,
                channels=6,
                dilations=(1, 2),
            ),
        ),
        (
            AutoregressiveTransformer(
                CharTokenizer.from_text(["abc"]),
                d_model=8,
                nhead=2,
                num_layers=2,
                dim_feedforward=12,
                max_length=8,
            ),
            AutoregressiveTransformerConfig(
                d_model=8,
                nhead=2,
                num_layers=2,
                dim_feedforward=12,
                max_length=8,
            ),
        ),
    ],
)
def test_load_inference_model_restores_cached_neural_models(tmp_path, model, config):
    """新输出层和缓存实现不改变独立 artifact 的加载约定。"""

    model_path = tmp_path / "model.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    config_path = tmp_path / "inference.json"
    torch.save(model.state_dict(), model_path)
    model.tokenizer.dump(tokenizer_path)
    save_inference_config(config_path, config)

    loaded, loaded_tokenizer = load_inference_model(
        model_path,
        tokenizer_path,
        config_path,
    )

    assert loaded_tokenizer == model.tokenizer
    assert type(loaded) is type(model)
    assert not loaded.training
    torch.testing.assert_close(loaded.state_dict(), model.state_dict())


def test_load_inference_model_rejects_legacy_tag(tmp_path):
    _, _, model_path, tokenizer_path, config_path = make_gru_artifact(tmp_path)
    config_path.write_text(json.dumps({"model_type": "mini_transformer"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_inference_model(model_path, tokenizer_path, config_path)


@pytest.mark.parametrize(
    "config",
    [
        AutoregressiveGRUConfig(embedding_dim=4, hidden_size=5),
        AutoregressiveTransformerConfig(d_model=4, nhead=2, num_layers=1),
    ],
)
def test_save_inference_config_round_trip(tmp_path, config):
    path = tmp_path / "inference.json"
    save_inference_config(path, config)
    assert json.loads(path.read_text()) == inference_config_dict(config)


def test_load_inference_model_restores_bigram_artifact(tmp_path):
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveBigram(tokenizer)
    model.count[model.tokenizer.bos_id, model.tokenizer.token_to_id["a"]] = 3
    model_path = tmp_path / "bigram.pt"
    model.save(model_path)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.dump(tokenizer_path)
    config_path = tmp_path / "inference.json"
    config_path.write_text(json.dumps({"model_type": "bigram", "alpha": 1.0}), encoding="utf-8")
    loaded, loaded_tokenizer = load_inference_model(model_path, tokenizer_path, config_path)
    assert loaded_tokenizer == tokenizer
    torch.testing.assert_close(loaded.count, model.count)


def test_score_passwords_uses_common_batch_interface():
    tokenizer = CharTokenizer.from_text(["ab", "abc"])
    model = AutoregressiveGRU(tokenizer, embedding_dim=4, hidden_size=5)
    passwords = ["abc", "", "ab", "?"]
    first = score_passwords(model, tokenizer, passwords, batch_size=1)
    second = score_passwords(model, tokenizer, passwords, batch_size=3)
    assert first == pytest.approx(second)
    assert score_password(model, tokenizer, "ab") == pytest.approx(first[2])


def test_score_passwords_verbose_and_empty(capsys):
    tokenizer = CharTokenizer.from_text(["a"])
    model = AutoregressiveBigram(tokenizer)
    assert score_passwords(model, tokenizer, []) == []
    score_passwords(model, tokenizer, ["a", "a"], batch_size=1, verbose=True)
    assert "completed=2/2" in capsys.readouterr().out


def test_score_passwords_rejects_invalid_arguments():
    tokenizer = CharTokenizer.from_text(["a"])
    model = AutoregressiveBigram(tokenizer)
    with pytest.raises(ValueError):
        score_passwords(model, tokenizer, ["a"], batch_size=0)
    with pytest.raises(ValueError):
        score_passwords(model, CharTokenizer.from_text(["b"]), ["a"])


class ScriptedModel(AutoregressiveGRU):
    def __init__(self, tokenizer, step_scores):
        super().__init__(tokenizer, embedding_dim=2, hidden_size=3)
        self.step_scores = step_scores
        self.step = 0

    def forward(self, input_ids, hidden=None, return_hidden=False):
        logits = torch.full((input_ids.size(0), input_ids.size(1), self.vocab_size), -torch.inf)
        for token_id, score in self.step_scores[min(self.step, len(self.step_scores) - 1)].items():
            logits[..., token_id] = score
        self.step += 1
        if hidden is None:
            hidden = torch.zeros(self.num_layers, input_ids.size(0), self.hidden_size)
        return (logits, hidden) if return_hidden else logits


def test_search_interfaces_are_model_agnostic_and_unique():
    tokenizer = CharTokenizer.from_text(["a", "b"])
    a_id = tokenizer.token_to_id["a"]
    b_id = tokenizer.token_to_id["b"]
    scores = [
        {a_id: 3.0, b_id: 2.0, tokenizer.eos_id: 1.0},
        {tokenizer.eos_id: 3.0, a_id: 1.0, b_id: 0.0},
    ]
    for search in (beam_search, best_first_search, depth_first_search):
        model = ScriptedModel(tokenizer, scores)
        kwargs = {"prefix": "", "beam_width": 2} if search is beam_search else {"num_candidates": 3}
        result = search(model, tokenizer, max_length=3, **kwargs)
        assert result
        assert len({candidate.text for candidate in result}) == len(result)
        assert all("PAD" not in candidate.text for candidate in result)


def test_search_accepts_bigram_and_tcn_models():
    tokenizer = CharTokenizer.from_text(["ab"])
    models = [
        AutoregressiveBigram(tokenizer),
        AutoregressiveTCN(tokenizer, embedding_dim=4, channels=4),
    ]
    for model in models:
        assert best_first_search(model, tokenizer, num_candidates=2, max_length=3)


def test_search_length_penalty_and_width_validation():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveBigram(tokenizer)
    with pytest.raises(ValueError):
        best_first_search(model, tokenizer, length_penalty=-1)
    with pytest.raises(ValueError):
        best_first_search(model, tokenizer, node_top_k=0)
    with pytest.raises(ValueError):
        best_first_search(model, tokenizer, depth_beam_width=0)


def test_search_persists_candidates(tmp_path):
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveBigram(tokenizer)
    path = tmp_path / "candidates.json"
    expected = best_first_search(model, tokenizer, num_candidates=3, max_length=3, save_path=path)
    actual = load_generation_candidates(path)
    assert actual == expected
    manual_path = tmp_path / "manual.json"
    values = [GenerationCandidate("a", [1], -1.0)]
    save_generation_candidates(manual_path, values)
    assert load_generation_candidates(manual_path) == values


def test_depth_first_verbose_is_silent_by_default(capsys):
    tokenizer = CharTokenizer.from_text(["a"])
    depth_first_search(AutoregressiveBigram(tokenizer), tokenizer, num_candidates=2, max_length=2)
    assert capsys.readouterr().out == ""
