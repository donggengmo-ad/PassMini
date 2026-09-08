import json
from dataclasses import asdict

import pytest
import torch

from scripts.experiment import (
    AutoregressiveGRUConfig,
    AutoregressiveMLPConfig,
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
    AutoregressiveMLP,
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
            AutoregressiveMLP(
                CharTokenizer.from_text(["abc"]),
                tau=3,
                embedding_dim=4,
                hidden_size=6,
            ),
            AutoregressiveMLPConfig(tau=3, embedding_dim=4, hidden_size=6),
        ),
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
        AutoregressiveMLPConfig(tau=2, embedding_dim=4, hidden_size=5),
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


def test_score_passwords_reports_realtime_progress():
    tokenizer = CharTokenizer.from_text(["a"])
    model = AutoregressiveBigram(tokenizer)
    updates = []

    score_passwords(
        model,
        tokenizer,
        ["a", "a", "a"],
        batch_size=2,
        progress_callback=lambda step, metrics: updates.append((step, dict(metrics))),
    )

    assert [step for step, _ in updates] == [2, 3]
    assert updates[-1][1]["progress"] == pytest.approx(1.0)
    assert updates[-1][1]["mean_surprisal_bits"] > 0
    assert updates[-1][1]["passwords_per_second"] > 0


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


class TrackingBigram(AutoregressiveBigram):
    """记录搜索每次模型前推实际使用的 batch size。"""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.advance_batch_sizes = []

    def advance_state(self, state, token_ids):
        self.advance_batch_sizes.append(token_ids.numel())
        return super().advance_state(state, token_ids)


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


def test_search_accepts_bigram_mlp_and_tcn_models():
    tokenizer = CharTokenizer.from_text(["ab"])
    models = [
        AutoregressiveBigram(tokenizer),
        AutoregressiveMLP(tokenizer, tau=2, embedding_dim=4, hidden_size=5),
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
    with pytest.raises(ValueError):
        best_first_search(model, tokenizer, expansion_batch_size=0)
    with pytest.raises(TypeError):
        best_first_search(model, tokenizer, use_cache=1)


def test_beam_search_advances_retained_frontier_as_one_batch():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = TrackingBigram(tokenizer)

    candidates = beam_search(model, tokenizer, prefix="", beam_width=3, max_length=4)

    assert len(candidates) == 3
    assert model.advance_batch_sizes
    assert max(model.advance_batch_sizes) > 1
    assert len(model.advance_batch_sizes) <= 3


def test_strict_best_first_batches_sibling_model_inputs():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = TrackingBigram(tokenizer)

    candidates = best_first_search(
        model,
        tokenizer,
        num_candidates=4,
        max_length=4,
        node_top_k=3,
        expansion_batch_size=1,
    )

    assert len(candidates) == 4
    assert max(model.advance_batch_sizes) > 1


def test_best_first_batch_pop_is_explicitly_approximate():
    tokenizer = CharTokenizer.from_text(["ab"])
    a_id = tokenizer.token_to_id["a"]
    b_id = tokenizer.token_to_id["b"]
    model = AutoregressiveBigram(tokenizer)
    model.count[tokenizer.bos_id, a_id] = 800
    model.count[tokenizer.bos_id, b_id] = 150
    model.count[tokenizer.bos_id, tokenizer.eos_id] = 50
    model.count[a_id, tokenizer.eos_id] = 900

    strict = best_first_search(
        model,
        tokenizer,
        num_candidates=1,
        max_length=3,
        expansion_batch_size=1,
    )
    approximate = best_first_search(
        model,
        tokenizer,
        num_candidates=1,
        max_length=3,
        expansion_batch_size=3,
    )

    assert strict[0].text == "a"
    assert approximate[0].text == ""


def test_best_first_batch_pop_supports_transformer_depth_groups():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveTransformer(
        tokenizer,
        d_model=4,
        nhead=2,
        num_layers=1,
        dim_feedforward=8,
        max_length=4,
    )

    candidates = best_first_search(
        model,
        tokenizer,
        num_candidates=3,
        max_length=3,
        node_top_k=3,
        expansion_batch_size=8,
    )

    assert len(candidates) == 3


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            AutoregressiveTCN(
                CharTokenizer.from_text(["ab"]),
                embedding_dim=4,
                channels=4,
            ),
            id="tcn",
        ),
        pytest.param(
            AutoregressiveTransformer(
                CharTokenizer.from_text(["ab"]),
                d_model=4,
                nhead=2,
                num_layers=1,
                dim_feedforward=8,
                max_length=4,
            ),
            id="transformer",
        ),
    ],
)
def test_best_first_without_cache_matches_cached_search(model):
    tokenizer = model.tokenizer

    cached = best_first_search(
        model,
        tokenizer,
        num_candidates=4,
        max_length=3,
        node_top_k=3,
        expansion_batch_size=1,
    )
    uncached = best_first_search(
        model,
        tokenizer,
        num_candidates=4,
        max_length=3,
        node_top_k=3,
        expansion_batch_size=1,
        use_cache=False,
    )

    assert [candidate.text for candidate in uncached] == [
        candidate.text for candidate in cached
    ]
    assert [candidate.log_probability for candidate in uncached] == pytest.approx(
        [candidate.log_probability for candidate in cached],
        abs=1e-5,
    )


def test_best_first_without_cache_bypasses_incremental_state(monkeypatch):
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveTCN(tokenizer, embedding_dim=4, channels=4)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("无缓存搜索不应创建或推进增量状态")

    monkeypatch.setattr(model, "initial_state", fail_if_called)
    monkeypatch.setattr(model, "advance_state", fail_if_called)

    candidates = best_first_search(
        model,
        tokenizer,
        num_candidates=3,
        max_length=3,
        node_top_k=3,
        use_cache=False,
    )

    assert len(candidates) == 3


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


def test_best_first_search_reports_final_progress():
    tokenizer = CharTokenizer.from_text(["ab"])
    updates = []

    candidates = best_first_search(
        AutoregressiveBigram(tokenizer),
        tokenizer,
        num_candidates=3,
        max_length=3,
        progress_callback=lambda step, metrics: updates.append((step, dict(metrics))),
    )

    assert len(candidates) == 3
    assert updates
    assert updates[-1][0] > 0
    assert updates[-1][1]["candidate_progress"] == pytest.approx(1.0)
    assert updates[-1][1]["completed_candidates"] == 3


def test_depth_first_verbose_is_silent_by_default(capsys):
    tokenizer = CharTokenizer.from_text(["a"])
    depth_first_search(AutoregressiveBigram(tokenizer), tokenizer, num_candidates=2, max_length=2)
    assert capsys.readouterr().out == ""
