"""全面检查中发现的跨模块行为回归。"""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.data import PasswordDataset, count_lines, read_dataset
from scripts.evaluation import (
    coverage_curve,
    random_generation_coverage_curve,
    save_surprisal_npz,
)
from scripts.inference import GenerationCandidate, beam_search, best_first_search
from scripts.models import AutoregressiveBigram, AutoregressiveGRU
from scripts.tokenizer import CharTokenizer
from app.frontend.playground import trace_character_probabilities


def test_bigram_batched_scores_match_transition_reference_and_mutations():
    tokenizer = CharTokenizer.from_text(["abc"])
    model = AutoregressiveBigram(tokenizer, alpha=0.25)
    model.fit(PasswordDataset(["ab", "ab", "c"], tokenizer))
    texts = ["abc", "", "a", "?", "cc"]
    for _ in range(2):
        reference = []
        for text in texts:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            reference.append(sum(
                math.log((int(model.count[a, b]) + model.alpha)
                         / (int(model.count[a].sum()) + model.vocab_size * model.alpha))
                for a, b in zip(ids, ids[1:])
            ))
        torch.testing.assert_close(model.log_probabilities(texts), torch.tensor(reference))
        model.count[tokenizer.bos_id, tokenizer.eos_id] += 20
    assert model.log_probabilities([]).shape == (0,)


def test_bigram_search_cache_modes_return_same_candidates():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveBigram(tokenizer)
    model.fit(PasswordDataset(["a", "ab", "ba"], tokenizer))
    kwargs = dict(num_candidates=10, max_length=3)
    cached = best_first_search(model, tokenizer, use_cache=True, **kwargs)
    plain = best_first_search(model, tokenizer, use_cache=False, **kwargs)
    assert [x.text for x in cached] == [x.text for x in plain]
    assert [x.log_probability for x in cached] == pytest.approx(
        [x.log_probability for x in plain]
    )


def test_gru_scoring_does_not_consume_sampling_rng():
    model = AutoregressiveGRU(CharTokenizer.from_text(["ab"]), 4, 6).eval()
    before = torch.random.get_rng_state().clone()
    model.log_probabilities(["ab", "a", ""])
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_generation_rejects_invalid_logits_explicitly(bad_value):
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveBigram(tokenizer)
    logits = torch.zeros(1, tokenizer.vocab_size)
    logits[0, tokenizer.token_to_id["a"]] = bad_value
    with pytest.raises(ValueError, match="NaN|无穷"):
        model._next_token_probs(logits, 1.0)


def test_beam_rejects_unknown_prefix_instead_of_silently_dropping_it():
    tokenizer = CharTokenizer.from_text(["ab"])
    with pytest.raises(ValueError, match="词表"):
        beam_search(AutoregressiveBigram(tokenizer), tokenizer, prefix="z")


def test_character_trace_uses_finite_log_scores_when_probability_underflows():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveBigram(tokenizer)
    logits = torch.zeros(1, 2, tokenizer.vocab_size)
    logits[:, :, tokenizer.token_to_id["a"]] = -1000
    with patch.object(model, "forward", return_value=logits) as forward:
        trace = trace_character_probabilities("a", {"model": (model, tokenizer)})[0]
    assert forward.call_count == 1
    assert trace.characters[0].probability == 0.0
    assert math.isfinite(trace.characters[0].surprisal_bits)
    assert trace.characters[0].surprisal_bits > 1400
    assert {x.token for x in trace.distribution} == {"a", "b", "[EOS]"}


def test_character_full_forward_matches_incremental_reference():
    tokenizer = CharTokenizer.from_text(["abc"])
    model = AutoregressiveGRU(tokenizer, 4, 6).eval()
    with torch.inference_mode():
        state = model.initial_state(1)
        reference = []
        for token in tokenizer.encode("abc"):
            reference.append(float(model._next_token_probs(state.next_logits, 1.0)[0, token]))
            state = model.advance_state(state, torch.tensor([token]))
    actual = trace_character_probabilities("abc", {"gru": (model, tokenizer)})[0]
    assert [x.probability for x in actual.characters] == pytest.approx(reference)


def test_read_dataset_limit_does_not_read_past_requested_lines(tmp_path):
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.txt").write_text("a\nb\nc\n")
    import scripts.data as data
    real_open = open

    class LimitedReader:
        def __init__(self, *args, **kwargs):
            self.stream = real_open(*args, **kwargs)
            self.read_count = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def __iter__(self):
            return self
        def __next__(self):
            self.read_count += 1
            assert self.read_count <= 1, "不应读取限额外的行"
            return next(self.stream)

    with patch.object(data, "open", LimitedReader, create=True):
        assert read_dataset(tmp_path, 1, 1, 1) == {
            "train": ["a"], "val": ["a"], "test": ["a"]
        }
    empty = tmp_path / "empty.txt"
    empty.touch()
    assert count_lines(empty) == 0


def test_npz_failed_export_preserves_previous_complete_file(tmp_path):
    path = tmp_path / "surprisal.npz"
    save_surprisal_npz(path, [2, 4], [1, 2])
    before = path.read_bytes()
    def interrupted(stream, **kwargs):
        stream.write(b"partial")
        raise OSError("simulated interrupted write")
    with patch("scripts.evaluation.np.savez_compressed", side_effect=interrupted):
        with pytest.raises(OSError):
            save_surprisal_npz(path, [9], [3])
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_coverage_retains_partial_and_sub_interval_final_budget():
    for count in (1, 3):
        texts = ["a", "b", "c"][:count]
        candidates = [GenerationCandidate(x, [], -1) for x in texts]
        for points in (
            coverage_curve(candidates, ["a", "b", "c"], checkpoint_step=2),
            random_generation_coverage_curve(texts, ["a", "b", "c"], checkpoint_step=2),
        ):
            assert points[-1].attempts == count
            assert points[-1].coverage == pytest.approx(count / 3)


@pytest.mark.parametrize("tokens", [["a"], ["BOS", "PAD", "EOS", "UNK"],
                                     ["PAD", "BOS", "EOS", "UNK", "a", "a"]])
def test_tokenizer_rejects_corrupt_mapping(tokens):
    with pytest.raises(ValueError):
        CharTokenizer(tokens)


@pytest.mark.parametrize("mismatch", ["tokenizer", "model_config", "scheduler"])
def test_resume_rejects_semantic_mismatch_before_overwriting_weights(tmp_path, mismatch):
    from torch.utils.data import DataLoader
    from scripts.training import save_checkpoint, train
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveGRU(tokenizer, 4, 6)
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(tmp_path / "checkpoint_latest.pt", model, optimizer, epoch=0)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    kwargs = {}
    if mismatch == "tokenizer":
        model.tokenizer = CharTokenizer([*CharTokenizer.SPECIAL_TOKENS, "b", "a"])
    elif mismatch == "model_config":
        kwargs["model_config"] = {"model_type": "tcn"}
    else:
        kwargs["scheduler"] = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loader = DataLoader(PasswordDataset(["ab"], model.tokenizer), batch_size=1)
    with pytest.raises(ValueError, match=mismatch):
        train(model, loader, loader, optimizer, resume=True, save_path=tmp_path, **kwargs)
    torch.testing.assert_close(before, model.state_dict())


def test_training_rejects_empty_validation_loader():
    from torch.utils.data import DataLoader
    from scripts.training import train
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveGRU(tokenizer, 4, 6)
    optimizer = torch.optim.Adam(model.parameters())
    full = DataLoader(PasswordDataset(["ab"], tokenizer))
    empty = DataLoader(PasswordDataset([], tokenizer))
    with pytest.raises(ValueError, match="batch"):
        train(model, full, empty, optimizer)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_configuration_rejects_nonfinite_training_values(bad):
    from scripts.experiment import TrainingConfig, SchedulerConfig, AutoregressiveBigramConfig
    for constructor in (
        lambda: TrainingConfig(learning_rate=bad),
        lambda: TrainingConfig(max_norm=bad),
        lambda: SchedulerConfig(gamma=bad),
        lambda: AutoregressiveBigramConfig(alpha=bad),
    ):
        with pytest.raises(ValueError, match="有限"):
            constructor()


def test_overview_does_not_extrapolate_unfinished_coverage():
    from app.frontend.library import CoverageData, coverage_at_budget
    data = CoverageData(100, np.array([100, 200]), np.array([0.1, 0.2]))
    assert coverage_at_budget(data, 201) is None
    assert coverage_at_budget(data, 200) == pytest.approx(0.2)
