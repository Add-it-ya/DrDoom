"""Build the datasets every later stage reads, and a manifest describing them.

Run with ``python -m drdoom.data.build``.

The real dataset already separates an anomaly-free period from a labelled one, so its
own train split becomes the training data and its labelled split is divided into
validation and test. The synthetic corpus has no such structure and is partitioned by
whichever strategy is requested.

The manifest is generated, never hand-written. Event counts per split per class are the
numbers that decide whether any downstream metric means anything, so they are recorded
by the code that produced them rather than transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from drdoom.config import get_settings
from drdoom.data import smd, store, synthetic, windows
from drdoom.data.schema import MetricSeries
from drdoom.data.splits import SplitResult, held_out_series_split, slice_series, time_based_split

logger = logging.getLogger(__name__)

SOURCES = ("smd", "synthetic")
STRATEGIES = ("time_based", "held_out_series")


def default_fractions(source: str, strategy: str) -> tuple[float, float]:
    """Split fractions that leave a usefully sized test set for each combination.

    Holding out whole machines from a 28 machine dataset at the usual 70/15/15 leaves
    five test machines and around forty events, which is too few to separate detectors
    with any confidence. Training data here is anomaly-free telemetry, and fifteen
    machines still supply several hundred thousand normal timesteps, so the real dataset
    trades training machines for test events.
    """
    if source == "smd" and strategy == "held_out_series":
        return 0.55, 0.15
    return 0.7, 0.15


@dataclass(frozen=True)
class BuildConfig:
    source: str
    strategy: str
    window_size: int = windows.DEFAULT_WINDOW
    stride: int = windows.DEFAULT_STRIDE
    n_scenarios: int = 200
    days: int = 4
    seed: int = 42
    train_frac: float = 0.7
    val_frac: float = 0.15
    output_root: Path | None = None

    @property
    def output_dir(self) -> Path:
        root = self.output_root or get_settings().processed_data_dir
        return root / self.source / self.strategy


def build_synthetic_splits(config: BuildConfig) -> SplitResult:
    series = synthetic.generate(n_scenarios=config.n_scenarios, days=config.days, seed=config.seed)
    if config.strategy == "time_based":
        return time_based_split(series, config.train_frac, config.val_frac)
    return held_out_series_split(series, config.train_frac, config.val_frac, seed=config.seed)


def build_smd_splits(config: BuildConfig) -> SplitResult:
    """Compose splits from the dataset's own anomaly-free and labelled periods."""
    loaded = smd.load_all()

    if config.strategy == "time_based":
        train = [train for train, _ in loaded.values()]
        val: list[MetricSeries] = []
        test: list[MetricSeries] = []
        for _, labelled in loaded.values():
            midpoint = labelled.n_timesteps // 2
            val.append(slice_series(labelled, 0, midpoint))
            test.append(slice_series(labelled, midpoint, labelled.n_timesteps))
        return SplitResult(strategy=config.strategy, train=train, val=val, test=test)

    machines = sorted(loaded)
    partition = held_out_series_split(
        [loaded[machine][1] for machine in machines],
        config.train_frac,
        config.val_frac,
        seed=config.seed,
    )
    val_ids = {item.series_id for item in partition.val}
    test_ids = {item.series_id for item in partition.test}
    train = [loaded[machine][0] for machine in machines if machine not in val_ids | test_ids]
    return SplitResult(
        strategy=config.strategy, train=train, val=partition.val, test=partition.test
    )


def summarise(series: list[MetricSeries], index: windows.WindowIndex) -> dict:
    events = [event for item in series for event in item.events]
    return {
        "series": len(series),
        "timesteps": sum(item.n_timesteps for item in series),
        "events": len(events),
        "events_by_label": dict(sorted(Counter(event.label for event in events).items())),
        "median_event_length": (
            int(sorted(event.length for event in events)[len(events) // 2]) if events else 0
        ),
        "windows": len(index),
        "anomaly_window_rate": round(index.anomaly_rate, 4),
    }


def build(config: BuildConfig) -> dict:
    """Produce the split datasets, the fitted scaler and the manifest."""
    logger.info("building %s with the %s strategy", config.source, config.strategy)
    result = build_smd_splits(config) if config.source == "smd" else build_synthetic_splits(config)

    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    scaler = windows.Scaler.fit(result.train)
    scaler.save(output / "scaler.npz")

    manifest: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": config.source,
        "strategy": config.strategy,
        "window_size": config.window_size,
        "stride": config.stride,
        "train_frac": config.train_frac,
        "val_frac": config.val_frac,
        "feature_names": list(result.train[0].feature_names),
        "splits": {},
    }

    for name, series in result.as_dict().items():
        index = windows.build_index(series, config.window_size, config.stride)
        store.save_series(output / f"{name}", series)
        _save_index(output / f"{name}_windows.npz", index)
        manifest["splits"][name] = summarise(series, index)

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("wrote %s", output / "manifest.json")
    return manifest


def _save_index(path: Path, index: windows.WindowIndex) -> None:
    np.savez(
        path,
        series_index=index.series_index,
        start=index.start,
        label=index.label,
        event_id=index.event_id,
        window_size=np.int32(index.window_size),
        stride=np.int32(index.stride),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=[*SOURCES, "both"], default="both")
    parser.add_argument("--strategy", choices=[*STRATEGIES, "both"], default="both")
    parser.add_argument("--window-size", type=int, default=windows.DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=windows.DEFAULT_STRIDE)
    parser.add_argument("--n-scenarios", type=int, default=200)
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--val-frac", type=float, default=None)
    parser.add_argument("--download", action="store_true", help="fetch the dataset if missing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    sources = SOURCES if args.source == "both" else (args.source,)
    strategies = STRATEGIES if args.strategy == "both" else (args.strategy,)

    if "smd" in sources and (args.download or not smd.is_downloaded()):
        smd.download()

    for source in sources:
        for strategy in strategies:
            train_frac, val_frac = default_fractions(source, strategy)
            config = BuildConfig(
                source=source,
                strategy=strategy,
                window_size=args.window_size,
                stride=args.stride,
                n_scenarios=args.n_scenarios,
                days=args.days,
                seed=args.seed,
                train_frac=args.train_frac if args.train_frac is not None else train_frac,
                val_frac=args.val_frac if args.val_frac is not None else val_frac,
            )
            manifest = build(config)
            counts = {name: split["events"] for name, split in manifest["splits"].items()}
            logger.info("%s/%s events per split: %s", source, strategy, counts)


if __name__ == "__main__":
    main()
