"""Compare every detector on every split, and write the results table.

Run with ``python -m drdoom.detect.compare``.

The table is generated, not typed. Whatever it says is what gets published: if a window
standard deviation beats the autoencoder, that is the finding, and the simple detector is
the one worth shipping.

Thresholds are chosen on validation against a false alarm budget and then applied
unchanged to test, so no detector gets to tune on the split it is scored on. The budget is
in pages as a person receives them, with a standing alarm paging again every hour.

Which detector is best is decided on validation, by the area under its detection-versus-
pages curve, never by test detection at one threshold. Picking the winner by the number it
is then judged on is how an earlier version of this table shipped a detector whose lead
came from how pages were counted.
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
INCUMBENT = "window_spread"


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
    reports: dict[str, ev.DetectionReport] = {}
    for detector in detectors:
        val_scores = detector.score(val_split.series, val_split.index)
        test_scores = detector.score(test_split.series, test_split.index)
        threshold = ev.select_threshold(
            val_scores, val_split.index, val_split.series, budget_per_day=budget_per_day
        )
        report = ev.evaluate(
            detector.name, test_scores, test_split.index, test_split.series, threshold
        )
        reports[detector.name] = report
        val_curve = ev.curve_auc(ev.detection_curve(val_scores, val_split.index, val_split.series))
        row = report.as_row() | {
            "source": source,
            "strategy": strategy,
            "val_curve_auc": round(val_curve, 4),
        }
        rows.append(row)
        logger.info(
            "%-32s detection %.3f (fresh %.3f)  pages/day %.2f  duty %.3f  curve %.3f",
            f"{source}/{strategy}/{detector.name}",
            row["detection_rate"],
            row["fresh_detection_rate"],
            row["pages_per_day"],
            row["alarm_duty"],
            row["curve_auc"],
        )

    incumbent = reports.get(INCUMBENT)
    for row in rows:
        if incumbent is None or row["detector"] == INCUMBENT:
            row["delta_vs_incumbent"] = None
            continue
        delta, interval = ev.paired_delta(reports[row["detector"]].outcomes, incumbent.outcomes)
        row["delta_vs_incumbent"] = round(delta, 4)
        row["delta_vs_incumbent_ci"] = [round(value, 4) for value in interval]
    return rows


def _interval(values: list[float] | None) -> str:
    return f"[{values[0]:.2f}, {values[1]:.2f}]" if values else "n/a"


def _is_autoencoder(row: dict) -> bool:
    return row["detector"].startswith("lstm_autoencoder")


def _chosen(candidates: list[dict]) -> dict:
    """The candidate validation prefers: highest curve area, then earliest detection."""
    return max(
        candidates,
        key=lambda r: (r.get("val_curve_auc", 0.0), -(r["median_minutes_to_detect"] or 1e9)),
    )


def _describe(row: dict) -> str:
    return (
        f"`{row['detector']}` (test curve {row['curve_auc']:.3f}; detection "
        f"{row['detection_rate']:.3f}, {row['fresh_detection_rate']:.3f} fresh, at "
        f"{row['pages_per_day']:.2f} pages/day with {row['alarm_duty']:.1%} of normal time "
        "in alarm)"
    )


def verdict_lines(rows: list[dict]) -> list[str]:
    """State, per split, which detector validation prefers and how it did on test.

    The choice is made on validation curve area, so the test numbers that follow are a
    check on that choice rather than the basis for it.
    """
    lines = [
        "## What the table says",
        "",
        "The autoencoder was built after the baselines specifically so this comparison could",
        "be made. Each line below is generated from the table, not asserted. The better",
        "detector is the one with the larger area under its validation detection-versus-pages",
        "curve; its test figures follow as a check.",
        "",
    ]
    for source in SOURCES:
        for strategy in STRATEGIES:
            subset = [r for r in rows if r["source"] == source and r["strategy"] == strategy]
            simple = [r for r in subset if not _is_autoencoder(r)]
            learned = [r for r in subset if _is_autoencoder(r)]
            if not simple or not learned:
                continue
            best_simple = _chosen(simple)
            best_learned = _chosen(learned)
            margin = best_learned["val_curve_auc"] - best_simple["val_curve_auc"]

            if margin > 0:
                call = f"validation prefers the autoencoder ({margin:+.3f} curve area)"
            elif margin < 0:
                call = f"validation prefers `{best_simple['detector']}` ({margin:+.3f} curve area)"
            else:
                call = "validation cannot separate them"

            lines.append(
                f"- **{source} / {strategy}**: best without a network is "
                f"{_describe(best_simple)}; the best autoencoder is {_describe(best_learned)}. "
                f"Here {call}."
            )

    lines += _standing_alarm_lines(rows)

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


def _standing_alarm_lines(rows: list[dict]) -> list[str]:
    """Name detectors whose detections mostly come from an alarm that was already up."""
    heavy = [
        r
        for r in rows
        if r["detection_rate"] > 0
        and (r["pre_alarmed_share"] >= 0.2 or r["worst_alarm_duty"] >= 0.5)
    ]
    if not heavy:
        return []
    lines = [
        "",
        "### Standing alarms",
        "",
        "These detectors were already in alarm before a fifth or more of the incidents they",
        "caught, or spent half or more of some machine's normal time in alarm. Their detection",
        "rate overstates what a pager would have told anyone:",
        "",
    ]
    lines += [
        f"- {r['source']}/{r['strategy']} `{r['detector']}`: {r['pre_alarmed_share']:.0%} of "
        f"detections pre-alarmed; worst machine in alarm {r['worst_alarm_duty']:.0%} of the time"
        for r in heavy
    ]
    return lines


def _delta(row: dict) -> str:
    if row.get("delta_vs_incumbent") is None:
        return "—"
    low, high = row["delta_vs_incumbent_ci"]
    return f"{row['delta_vs_incumbent']:+.3f} [{low:+.3f}, {high:+.3f}]"


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
        "**Pages per day** counts what reaches a human for nothing: a run of consecutive",
        "firing windows that overlaps no incident is one page, plus one more for every hour",
        "it stays up, as alerting re-notifies an unresolved alert. *Episodes/day* is the",
        "older count, one page per run however long it lasts, kept for comparison. It",
        "rewards a detector that never switches off.",
        "",
        "**Fresh** detection leaves out incidents that were already covered by an alarm",
        "raised before they began. **In alarm** is the share of normal time spent alarming.",
        "",
        "**Curve** is the mean, over budgets from 0.5 to 4 pages per series-day, of the best",
        "detection rate each budget allows: one number per detector that does not depend on",
        f"a single threshold. **Δ vs `{INCUMBENT}`** is the paired difference in detection",
        "rate over the same incidents, with a 95% bootstrap interval.",
        "",
        f"Thresholds were chosen on validation at a budget of {budget:g} page(s) per",
        "series-day, re-paging included, and applied unchanged to test. Intervals are 95%",
        "percentile bootstrap, resampled over events for detection and over series for",
        "page rate.",
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
                "| Detector | Curve | Detection | 95% CI | Fresh | Minutes to detect |"
                " Pages/day | Episodes/day | In alarm | Δ vs incumbent | PR-AUC |",
                "|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|",
            ]
            for row in sorted(subset, key=lambda r: -r["curve_auc"]):
                ttd = row["median_minutes_to_detect"]
                lines.append(
                    f"| {row['detector']} | {row['curve_auc']:.3f} |"
                    f" {row['detection_rate']:.3f} |"
                    f" {_interval(row['detection_rate_ci'])} |"
                    f" {row['fresh_detection_rate']:.3f} |"
                    f" {ttd if ttd is not None else 'n/a'} |"
                    f" {row['pages_per_day']:.2f} |"
                    f" {row['false_alarms_per_day']:.2f} |"
                    f" {row['alarm_duty']:.1%} |"
                    f" {_delta(row)} |"
                    f" {row['pr_auc']:.3f} |"
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
    (docs / "detection-results.json").write_text(
        json.dumps({"budget": args.budget, "rows": rows}, indent=2), encoding="utf-8"
    )
    (docs / "detection-results.md").write_text(render_markdown(rows, args.budget), encoding="utf-8")
    logger.info("wrote %s", docs / "detection-results.md")


if __name__ == "__main__":
    main()
