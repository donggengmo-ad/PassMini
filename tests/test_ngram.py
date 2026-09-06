from re import Pattern

from scripts.models import AutoregressiveBigram
from scripts.data import PasswordDataset
from scripts.tokenizer import CharTokenizer
from scripts.util import filtered_probs_with_temperature
import torch, pytest, re, math


def _reference_count(dataset: PasswordDataset) -> torch.Tensor:
    count = torch.zeros(
        (dataset.tokenizer.vocab_size, dataset.tokenizer.vocab_size),
        dtype=torch.long,
    )
    for input_ids, target_ids in dataset:
        for input_id, target_id in zip(input_ids, target_ids):
            count[input_id, target_id] += 1
    return count

def test_bigram_init():
    token = CharTokenizer.from_text(['abcd'])
    dataset = PasswordDataset(['ab', 'ac'], token)
    vocab_size = token.vocab_size
    model = AutoregressiveBigram(tokenizer=token)
    model.fit(dataset)
    assert model.count.shape == (vocab_size, vocab_size)
    prob = lambda x, y: model.count[token.token_to_id[x], token.token_to_id[y]]
    assert prob(token.BOS_TOKEN, 'a') == 2
    assert prob('a', 'b') == 1
    assert prob('a', 'c') == 1
    assert prob('b', token.EOS_TOKEN) == 1
    assert prob('c', token.EOS_TOKEN) == 1

    probs = model.next_token_probs(token.token_to_id['a'])
    assert probs.shape == torch.Size([vocab_size])
    assert probs.dtype == torch.float32
    torch.testing.assert_close(probs.sum(),torch.tensor(1.0, dtype=probs.dtype))
    assert torch.all(probs >= 0)
    torch.testing.assert_close(probs[token.token_to_id['b']], probs[token.token_to_id['c']])
    processed_probs = filtered_probs_with_temperature(
        token,
        probs,
        temperature=1.0,
    )
    torch.testing.assert_close(processed_probs[token.bos_id], torch.tensor(0.0, dtype=processed_probs.dtype))
    torch.testing.assert_close(processed_probs[token.pad_id], torch.tensor(0.0, dtype=processed_probs.dtype))
    torch.testing.assert_close(processed_probs[token.unk_id], torch.tensor(0.0, dtype=processed_probs.dtype))
    torch.testing.assert_close(processed_probs.sum(), torch.tensor(1.0, dtype=processed_probs.dtype))
    assert processed_probs[token.eos_id] > 0
    assert torch.all(torch.isfinite(processed_probs))
    count_bak = model.count.clone()
    model.fit(dataset)
    torch.testing.assert_close(count_bak, model.count)
    with pytest.raises(ValueError):
        AutoregressiveBigram(tokenizer=token, alpha=0)

def test_bigram_log_probs():
    word_list = ['ab', 'ac']
    token = CharTokenizer.from_text(word_list)
    dataset = PasswordDataset(word_list, token)
    model = AutoregressiveBigram(tokenizer=token)
    model.fit(dataset)
    text = 'ab'
    log_prob = model.log_probability(text)
    assert isinstance(log_prob, torch.Tensor)
    assert log_prob.shape == torch.Size([])
    # bos->a:1/3; a->b:2/9; b->eos:1/4
    torch.testing.assert_close(log_prob, torch.tensor(-3.9889840465642745, dtype=log_prob.dtype))
    surprise = model.surprisal_bits(text)
    assert isinstance(surprise, torch.Tensor)
    assert surprise.shape == torch.Size([])
    torch.testing.assert_close(surprise, torch.tensor(5.754887502163469, dtype=log_prob.dtype))

def test_bigram_generate():
    word_list = ['this', 'is', 'a', 'test', 'for', 'bigram', 'language', 'model', 'with', 'a', 'few', 'words']
    token = CharTokenizer.from_text(word_list)
    dataset = PasswordDataset(word_list, token)
    model = AutoregressiveBigram(tokenizer=token)
    model.fit(dataset)
    with pytest.raises(ValueError):
        model.generate(max_length=10, temperature=-1)
    with pytest.raises(ValueError):
        model.generate(max_length=-10, temperature=1)
    for _ in range(10):
        max_length = math.floor(torch.rand(1).item() * 10 + 1)
        generated = model.generate(max_length=max_length)
        assert isinstance(generated, str)
        assert len(generated) <= max_length
        assert [re.findall(spe, generated) for spe in token.SPECIAL_TOKENS] == [[], [], [], []]


def test_bigram_generate_batch_returns_requested_count_and_valid_lengths():
    word_list = ["a", "ab", "ba"]
    tokenizer = CharTokenizer.from_text(word_list)
    model = AutoregressiveBigram(tokenizer=tokenizer)
    model.fit(PasswordDataset(word_list, tokenizer))

    generated = model.generate_batch(batch_size=5, max_length=4)

    assert len(generated) == 5
    assert all(isinstance(password, str) for password in generated)
    assert all(len(password) <= 4 for password in generated)
    assert all(set(password) <= {"a", "b"} for password in generated)


def test_bigram_generate_batch_is_reproducible_with_same_generator_seed():
    tokenizer = CharTokenizer.from_text(["ab"])
    model = AutoregressiveBigram(tokenizer=tokenizer)

    first = model.generate_batch(
        batch_size=4,
        max_length=5,
        generator=torch.Generator().manual_seed(42),
    )
    second = model.generate_batch(
        batch_size=4,
        max_length=5,
        generator=torch.Generator().manual_seed(42),
    )

    assert first == second


@pytest.mark.parametrize("batch_size", [0, -1])
def test_bigram_generate_batch_rejects_non_positive_batch_size(batch_size):
    model = AutoregressiveBigram(tokenizer=CharTokenizer.from_text(["a"]))

    with pytest.raises(ValueError, match="batch_size"):
        model.generate_batch(batch_size=batch_size, max_length=3)


def test_bigram_fit_matches_reference_count_with_batches():
    word_list = ["ab", "abc", "b", "cab"]
    tokenizer = CharTokenizer.from_text(word_list)
    dataset = PasswordDataset(word_list, tokenizer)
    model = AutoregressiveBigram(tokenizer=tokenizer)

    model.fit(dataset, batch_size=2)

    torch.testing.assert_close(model.count, _reference_count(dataset))


def test_bigram_fit_rejects_non_positive_batch_size():
    tokenizer = CharTokenizer.from_text(["ab"])
    dataset = PasswordDataset(["ab"], tokenizer)
    model = AutoregressiveBigram(tokenizer=tokenizer)

    with pytest.raises(ValueError, match="batch_size"):
        model.fit(dataset, batch_size=0)


def test_bigram_save_load_round_trip(tmp_path):
    word_list = ["ab", "abc", "b"]
    tokenizer = CharTokenizer.from_text(word_list)
    model = AutoregressiveBigram(tokenizer=tokenizer, alpha=0.25)
    model.fit(PasswordDataset(word_list, tokenizer), batch_size=2)
    path = tmp_path / "nested" / "bigram.pt"

    model.save(path)
    loaded = AutoregressiveBigram.load(path)

    assert loaded.tokenizer == model.tokenizer
    assert loaded.alpha == model.alpha
    torch.testing.assert_close(loaded.count, model.count)
    torch.testing.assert_close(
        loaded.next_token_probs(tokenizer.bos_id),
        model.next_token_probs(tokenizer.bos_id),
    )
    torch.testing.assert_close(
        loaded.log_probability("ab"),
        model.log_probability("ab"),
    )


def test_bigram_load_rejects_invalid_artifact(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save({"model_type": "other"}, path)

    with pytest.raises(ValueError, match="model_type"):
        AutoregressiveBigram.load(path)


def test_bigram_load_rejects_count_shape_mismatch(tmp_path):
    tokenizer = CharTokenizer.from_text(["ab"])
    path = tmp_path / "invalid.pt"
    torch.save(
        {
            "model_type": "bigram",
            "tokenizer": tokenizer.id_to_token,
            "alpha": 1.0,
            "count": torch.zeros((2, 2), dtype=torch.long),
        },
        path,
    )

    with pytest.raises(ValueError, match="count"):
        AutoregressiveBigram.load(path)
