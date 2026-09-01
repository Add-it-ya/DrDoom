"""The build produces usable splits and a manifest that describes them truthfully."""

import json

import pytest

from drdoom.data import smd
from drdoom.data.build import BuildConfig, build, default_fractions, parse_args


def synthetic_config(tmp_path, strategy: str = "time_based") -> BuildConfig:
    return BuildConfig(
        source="synthetic",
        strategy=strategy,
        n_scenarios=20,
        days=2,
        window_size=60,
        stride=30,
        output_root=tmp_path,
    )


def test_build_writes_every_expected_artefact(tmp_path) -> None:
    config = synthetic_config(tmp_path)

    build(config)

    output = config.output_dir
    assert (output / "manifest.json").is_file()
    assert (output / "scaler.npz").is_file()
    for split in ("train", "val", "test"):
        assert (output / f"{split}.npz").is_file()
        assert (output / f"{split}.json").is_file()
        assert (output / f"{split}_windows.npz").is_file()


def test_manifest_counts_match_the_data_on_disk(tmp_path) -> None:
    manifest = build(synthetic_config(tmp_path))

    written = json.loads(
        (synthetic_config(tmp_path).output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert written == manifest
    for split in manifest["splits"].values():
        assert split["events"] == sum(split["events_by_label"].values())
        assert split["windows"] > 0


def test_every_split_reports_every_root_cause(tmp_path) -> None:
    manifest = build(synthetic_config(tmp_path))

    for split in manifest["splits"].values():
        assert len(split["events_by_label"]) == 4


def test_held_out_strategy_uses_disjoint_series(tmp_path) -> None:
    manifest = build(synthetic_config(tmp_path, strategy="held_out_series"))

    counts = [split["series"] for split in manifest["splits"].values()]
    assert sum(counts) == 20


def test_real_dataset_held_out_split_trades_train_machines_for_test_events() -> None:
    assert default_fractions("smd", "held_out_series") == (0.55, 0.15)
    assert default_fractions("smd", "time_based") == (0.7, 0.15)
    assert default_fractions("synthetic", "held_out_series") == (0.7, 0.15)


def test_cli_defaults_cover_both_sources_and_strategies() -> None:
    args = parse_args([])

    assert args.source == "both"
    assert args.strategy == "both"
    assert args.train_frac is None


@pytest.mark.requires_dataset
def test_real_dataset_training_split_is_anomaly_free(tmp_path) -> None:
    if not smd.is_downloaded():
        pytest.skip("dataset not downloaded")

    manifest = build(
        BuildConfig(source="smd", strategy="time_based", stride=200, output_root=tmp_path)
    )

    assert manifest["splits"]["train"]["events"] == 0
    assert manifest["splits"]["train"]["anomaly_window_rate"] == 0.0
    assert manifest["splits"]["test"]["events"] > 100
