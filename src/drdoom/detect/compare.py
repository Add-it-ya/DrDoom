"""Compare every detector on every split, and write the results table.

Run with ``python -m drdoom.detect.compare``.

The table is generated, not typed. Whatever it says is what gets published: if a window
standard deviation beats the autoencoder, that is the finding, and the simple detector is
the one worth shipping.

Thresholds are chosen on validation against a false alarm budget and then applied
unchanged to test, so no detector gets to tune on the split it is scored on.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from drdoom.config import get_settings
from drdoom.data.windows import Scaler
from drdoom.detect import evaluate as ev
from drdoom.detect.autoencoder import AutoencoderDetector
from drdoom.detect.base import Detector
from drdoom.detect.baselines import all_baselines
from drdoom.detect.train import CRITERIA, TrainConfig, load_split, train

logger = logging.getLogger(__name__)

SOURCES = ("smd", "synthetic")
STRATEGIES = ("time_based", "held_out_series")


def autoencoder_for(config: TrainConfig, scaler: Scaler) -> AutoencoderDetector:
    """Load the trained checkpoint, training it first if it is not on disk."""
    if not config.model_path.is_file():
        logger.info("no checkpoint for %s, training now", config.criterion)
        train(config)
    detector = AutoencoderDetector.load(config.model_path, scaler)
    detector.name = f"lstm_autoencoder[{config.criterion}]"
    return detector


def run_split(
    source: str,
    strategy: str,
    budget_per_day: float,
    criteria: tuple[str, ...],
    max_epochs: int,
    max_train_windows: int,
    data_root: Path | None = None,
    models_root: Path | None = None,
) -> list[dict]:
    """Fit, threshold and score every detector for one source and strategy."""
    base = TrainConfig(
        source=source,
        strategy=strategy,
        max_epochs=max_epochs,
        max_train_windows=max_train_windows,
        data_root=data_root,
        models_root=models_root,
    )
    train_split = load_split(base, "train")
    val_split = load_split(base, "val")
    test_split = load_split(base, "test")
    scaler = Scaler.load(base.data_dir / "scaler.npz")
    train_normal = train_split.index.normal_only()

    detectors: list[Detector] = []
    for detector in all_baselines():
        detector.fit(train_split.series, train_normal, scaler)
        detectors.append(detector)
    for criterion in criteria:
        detectors.append(autoencoder_for(replace(base, criterion=criterion), scaler))

    rows = []
    for detector in detectors:
        val_scores = detector.score(val_split.series, val_split.index)
        test_scores = detector.score(test_split.series, test_split.index)
        threshold = ev.select_threshold(
            val_scores, val_split.index, val_split.series, budget_per_day=budget_per_day
        )
        report = ev.evaluate(
            detector.name, test_scores, test_split.index, test_split.series, threshold
        )
        row = report.as_row() | {"source": source, "strategy": strategy}
        rows.append(row)
        logger.info(
            "%-32s detection %.3f  ttd %s  alarms/day %.2f  pr_auc %.3f",
            f"{source}/{strategy}/{detector.name}",
            row["detection_rate"],
            row["median_minutes_to_detect"],
            row["false_alarms_per_day"],
            row["pr_auc"],
        )
    return rows


def _interval(values: list[float] | None) -> str:
    return f"[{values[0]:.2f}, {values[1]:.2f}]" if values else "n/a"


def _is_autoencoder(row: dict) -> bool:
    return row["detector"].startswith("lstm_autoencoder")


def verdict_lines(rows: list[dict]) -> list[str]:
    """State, per split, whether the network beat the best thing without one."""
    lines = [
        "## What the table says",
        "",
        "The autoencoder was built after the baselines specifically so this comparison could",
        "be made. Each line below is generated from the table, not asserted.",
        "",
    ]
    for source in SOURCES:
        for strategy in STRATEGIES:
            subset = [r for r in rows if r["source"] == source and r["strategy"] == strategy]
            simple = [r for r in subset if not _is_autoencoder(r)]
            learned = [r for r in subset if _is_autoencoder(r)]
            if not simple or not learned:
                continue
            best_simple = max(simple, key=lambda r: r["detection_rate"])
            best_learned = max(learned, key=lambda r: r["detection_rate"])
            margin = best_learned["detection_rate"] - best_simple["detection_rate"]

            if margin > 0:
                call = f"the autoencoder wins on detection ({margin:+.3f})"
            elif margin < 0:
                call = f"`{best_simple['detector']}` wins on detection ({margin:+.3f})"
            else:
                simple_delay = best_simple["median_minutes_to_detect"]
                learned_delay = best_learned["median_minutes_to_detect"]
                if simple_delay is not None and learned_delay is not None:
                    faster = "the autoencoder" if learned_delay < simple_delay else "the baseline"
                    call = (
                        f"detection ties, and {faster} is faster "
                        f"({learned_delay:g} against {simple_delay:g} minutes)"
                    )
                else:
                    call = "detection ties"

            lines.append(
                f"- **{source} / {strategy}**: best without a network is "
                f"`{best_simple['detector']}` at {best_simple['detection_rate']:.3f} detection, "
                f"{best_simple['false_alarms_per_day']:.2f} alarms/day; the autoencoder reaches "
                f"{best_learned['detection_rate']:.3f} at "
                f"{best_learned['false_alarms_per_day']:.2f} alarms/day. "
                f"Here {call}."
            )

    criteria_note: list[str] = []
    for source in SOURCES:
        for strategy in STRATEGIES:
            learned = [
                r
                for r in rows
                if r["source"] == source and r["strategy"] == strategy and _is_autoencoder(r)
            ]
            if len(learned) < 2:
                continue
            spread = max(r["detection_rate"] for r in learned) - min(
                r["detection_rate"] for r in learned
            )
            criteria_note.append(f"{source}/{strategy} {spread:+.3f}")

    if criteria_note:
        lines += [
            "",
            "### Checkpoint selection",
            "",
            "Three selection criteria were trained separately: reconstruction error over",
            "anomaly-free validation windows, ranking quality against validation labels, and",
            "reconstruction error over the whole validation split, anomalies included. The",
            "third contaminates the choice, because it rewards the model for reconstructing",
            "the anomalies it was deliberately never trained on.",
            "",
            "Spread in detection rate across the three, per split: "
            + ", ".join(criteria_note)
            + ".",
            "",
            "Where that spread is near zero the criterion did not matter in practice, which",
            "is the honest reading: the validation splits here carry few enough anomalies",
            "that the contaminated average stays close to the clean one. The clean criterion",
            "is still the default, because it costs nothing and does not depend on the",
            "anomaly rate staying low.",
        ]
    return [*lines, ""]


def render_markdown(rows: list[dict], budget: float) -> str:
    lines = [
        "# Detector comparison",
        "",
        "Generated by `python -m drdoom.detect.compare`. Nothing in this file is typed by",
        "hand, and the numbers are published whatever they say.",
        "",
        "## How to read it",
        "",
        "**Detection rate** is the fraction of distinct incidents where at least one",
        "overlapping window fired, not the fraction of windows classified correctly. A single",
        "outage lasting hours spans hundreds of overlapping windows, so window accuracy",
        "flatters a detector that catches one long incident and misses ten short ones.",
        "",
        "**Minutes to detect** is measured from the start of the incident to the end of the",
        "first window that fired, because a window cannot be scored until it is complete.",
        "",
        "**Alarms per day** counts runs of consecutive firing windows that overlap no",
        "incident as one page each, since that is what reaches a human.",
        "",
        f"Thresholds were chosen on validation at a budget of {budget:g} false alarm(s) per",
        "series-day and applied unchanged to test. Intervals are 95% percentile bootstrap,",
        "resampled over events for detection and over series for alarm rate.",
        "",
        "Results on the real dataset are **not** point-adjusted. Much of the published work on",
        "this benchmark credits an entire anomaly segment as detected whenever any single",
        "point inside it is flagged, which inflates F1 substantially and is not comparable to",
        "the numbers here. Detection rate below is the honest form of that idea: one incident,",
        "one outcome, counted once.",
        "",
    ]
    lines += verdict_lines(rows)

    for source in SOURCES:
        for strategy in STRATEGIES:
            subset = [r for r in rows if r["source"] == source and r["strategy"] == strategy]
            if not subset:
                continue
            events = subset[0]["events"]
            lines += [
                f"## {source} / {strategy}",
                "",
                f"{events} incidents in the test split.",
                "",
                "| Detector | Detection rate | 95% CI | Minutes to detect | Alarms/day |"
                " Window F1 | PR-AUC | ROC-AUC |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
            for row in sorted(subset, key=lambda r: -r["detection_rate"]):
                ttd = row["median_minutes_to_detect"]
                lines.append(
                    f"| {row['detector']} | {row['detection_rate']:.3f} |"
                    f" {_interval(row['detection_rate_ci'])} |"
                    f" {ttd if ttd is not None else 'n/a'} |"
                    f" {row['false_alarms_per_day']:.2f} |"
                    f" {row['window_f1']:.3f} | {row['pr_auc']:.3f} | {row['roc_auc']:.3f} |"
                )
            lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare detectors and write the results table.")
    parser.add_argument("--source", choices=[*SOURCES, "both"], default="both")
    parser.add_argument("--strategy", choices=[*STRATEGIES, "both"], default="both")
    parser.add_argument("--budget", type=float, default=ev.DEFAULT_FALSE_ALARM_BUDGET)
    parser.add_argument("--criteria", nargs="+", default=list(CRITERIA))
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--max-train-windows", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sources = SOURCES if args.source == "both" else (args.source,)
    strategies = STRATEGIES if args.strategy == "both" else (args.strategy,)

    rows: list[dict] = []
    for source in sources:
        for strategy in strategies:
            rows += run_split(
                source,
                strategy,
                args.budget,
                tuple(args.criteria),
                args.max_epochs,
                args.max_train_windows,
            )

    docs = args.out or get_settings().project_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "detection-results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (docs / "detection-results.md").write_text(render_markdown(rows, args.budget), encoding="utf-8")
    logger.info("wrote %s", docs / "detection-results.md")


if __name__ == "__main__":
    main()
