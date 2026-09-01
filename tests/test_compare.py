"""The results table must state what the numbers say, including when they disappoint."""

from drdoom.detect.compare import parse_args, render_markdown, verdict_lines


def row(detector: str, source: str, strategy: str, detection: float, delay: float = 10.0) -> dict:
    return {
        "detector": detector,
        "source": source,
        "strategy": strategy,
        "events": 100,
        "detection_rate": detection,
        "detection_rate_ci": [detection - 0.05, detection + 0.05],
        "median_minutes_to_detect": delay,
        "minutes_to_detect_ci": [delay - 1, delay + 1],
        "false_alarms_per_day": 1.0,
        "false_alarms_per_day_ci": [0.8, 1.2],
        "window_precision": 0.5,
        "window_recall": 0.5,
        "window_f1": 0.5,
        "roc_auc": 0.8,
        "pr_auc": 0.4,
        "threshold": 1.0,
    }


def test_verdict_names_the_baseline_when_it_wins() -> None:
    rows = [
        row("window_spread", "smd", "time_based", 0.78),
        row("lstm_autoencoder[normal_val_loss]", "smd", "time_based", 0.63),
    ]

    text = "\n".join(verdict_lines(rows))

    assert "`window_spread` wins on detection" in text
    assert "-0.150" in text


def test_verdict_names_the_network_when_it_wins() -> None:
    rows = [
        row("window_spread", "smd", "time_based", 0.60),
        row("lstm_autoencoder[normal_val_loss]", "smd", "time_based", 0.75),
    ]

    assert "the autoencoder wins on detection" in "\n".join(verdict_lines(rows))


def test_a_tie_is_broken_on_time_to_detect() -> None:
    rows = [
        row("window_spread", "synthetic", "time_based", 1.0, delay=19.0),
        row("lstm_autoencoder[normal_val_loss]", "synthetic", "time_based", 1.0, delay=16.0),
    ]

    text = "\n".join(verdict_lines(rows))

    assert "detection ties" in text
    assert "the autoencoder is faster" in text


def test_criterion_spread_is_reported_when_several_were_trained() -> None:
    rows = [
        row("window_spread", "smd", "time_based", 0.70),
        row("lstm_autoencoder[normal_val_loss]", "smd", "time_based", 0.63),
        row("lstm_autoencoder[mixed_val_loss]", "smd", "time_based", 0.60),
    ]

    text = "\n".join(verdict_lines(rows))

    assert "Checkpoint selection" in text
    assert "smd/time_based +0.030" in text


def test_verdict_is_skipped_when_there_is_nothing_to_compare() -> None:
    text = "\n".join(verdict_lines([row("window_spread", "smd", "time_based", 0.7)]))

    assert "wins on detection" not in text


def test_rendered_report_has_a_section_per_split() -> None:
    rows = [
        row("window_spread", "smd", "time_based", 0.78),
        row("lstm_autoencoder[normal_val_loss]", "smd", "time_based", 0.63),
        row("window_spread", "synthetic", "held_out_series", 1.0),
        row("lstm_autoencoder[normal_val_loss]", "synthetic", "held_out_series", 1.0),
    ]

    text = render_markdown(rows, budget=1.0)

    assert "## smd / time_based" in text
    assert "## synthetic / held_out_series" in text
    assert "100 incidents in the test split." in text


def test_report_warns_that_results_are_not_point_adjusted() -> None:
    text = render_markdown([row("window_spread", "smd", "time_based", 0.7)], budget=1.0)

    assert "not** point-adjusted" in text


def test_detectors_are_listed_best_first() -> None:
    rows = [
        row("weak", "smd", "time_based", 0.30),
        row("strong", "smd", "time_based", 0.90),
    ]

    text = render_markdown(rows, budget=1.0)

    assert text.index("| strong |") < text.index("| weak |")


def test_cli_defaults_cover_everything() -> None:
    args = parse_args([])

    assert args.source == "both"
    assert args.strategy == "both"
    assert len(args.criteria) == 3
