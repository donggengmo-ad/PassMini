import json

from utils.calculate_model_flops import (
    estimate_bigram_flops,
    estimate_gru_flops,
    estimate_mlp_flops,
    estimate_tcn_flops,
    estimate_transformer_flops,
    synchronize_flops,
)


def test_architecture_formulas_have_independent_small_expectations():
    assert estimate_bigram_flops(vocab_size=5, sequence_length=2) == 20
    assert estimate_mlp_flops(5, 2, 3, 2, 4) == 166
    assert estimate_gru_flops(5, 2, 2, 3, 1) == 294
    assert estimate_tcn_flops(5, 2, 2, 3, 2, 2) == 186
    assert estimate_transformer_flops(5, 2, 4, 2, 1, 6) == 736


def test_synchronize_flops_updates_tiers_and_frontend_catalog(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(["PAD", "BOS", "EOS", "UNK", "a"]), encoding="utf-8")

    tier_path = tmp_path / "output" / "model_tiers.json"
    tier_path.parent.mkdir()
    tier_path.write_text(
        json.dumps(
            {
                "low": {
                    "mlp": {
                        "model_type": "mlp",
                        "tau": 3,
                        "embedding_dim": 2,
                        "hidden_size": 4,
                        "parameter_count": 9,
                    },
                    "gru": {
                        "model_type": "gru",
                        "embedding_dim": 2,
                        "hidden_size": 3,
                        "num_layers": 1,
                        "parameter_count": 10,
                        "mult_adds": 999,
                    },
                    "tcn": {
                        "model_type": "tcn",
                        "embedding_dim": 2,
                        "channels": 3,
                        "kernel_size": 2,
                        "dilations": [1, 2],
                        "parameter_count": 11,
                    },
                    "transformer": {
                        "model_type": "transformer",
                        "d_model": 4,
                        "nhead": 2,
                        "num_layers": 1,
                        "dim_feedforward": 6,
                        "max_length": 4,
                        "parameter_count": 12,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    catalog_path = tmp_path / "app" / "artifacts" / "catalog.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "baseline-bigram",
                        "tier": "baseline",
                        "model_type": "bigram",
                        "parameter_count": 0,
                        "mult_adds": 0,
                    },
                    {
                        "id": "low-gru",
                        "tier": "low",
                        "model_type": "gru",
                        "parameter_count": 10,
                        "mult_adds": 999,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    results = synchronize_flops(
        project_root=tmp_path,
        tokenizer_path=tokenizer_path,
        catalog_path=catalog_path,
        tier_paths=[tier_path],
        sequence_length=2,
    )

    assert results[("low", "gru")] == 294
    assert results[("low", "mlp")] == 166
    assert results[("baseline", "bigram")] == 20
    tiers = json.loads(tier_path.read_text(encoding="utf-8"))
    assert tiers["low"]["gru"]["flops"] == 294
    assert tiers["low"]["mlp"]["flops"] == 166
    assert tiers["low"]["tcn"]["flops"] == 186
    assert tiers["low"]["transformer"]["flops"] == 736
    assert all(
        "mult_adds" not in config
        for config in tiers["low"].values()
    )

    records = json.loads(catalog_path.read_text(encoding="utf-8"))["models"]
    assert records[0]["flops"] == 20
    assert records[1]["flops"] == 294
    assert all("mult_adds" not in record for record in records)
