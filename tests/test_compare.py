"""The results table must state what the numbers say, including when they disappoint."""

from drdoom.detect.compare import parse_args, render_markdown, verdict_lines


def row(
    detector: str,
    source: str,
    strategy: str,
    detection: float,
    delay: float = 10.0,
    val_curve: float | None = None,
    pre_alarmed: float = 0.0,
    worst_duty: float = 0.05,
) -> dict:
    return {
        "detector": detector,
        "source": source,
        "strategy": strategy,
        "events": 100,
        "detection_rate": detection,
        "detection_rate_ci": [detection - 0.05, detection + 0.05],
        "fresh_detection_rate": detection * (1 - pre_alarmed),
        "pre_alarmed_share": pre_alarmed,
        "median_minutes_to_detect": delay,
        "minutes_to_detect_ci": [delay - 1, delay + 1],
        "false_alarms_per_day": 1.0,
        "false_alarms_per_day_ci": [0.8, 1.2],
        "pages_per_day": 1.5,
        "pages_per_day_ci": [1.2, 1.8],
        "alarm_duty": 0.02,
        "worst_alarm_duty": worst_duty,
        "curve_auc": detection,
        "val_curve_auc": detection if val_curve is None else val_curve,
        "delta_vs_incumbent": None,
        "window_precision": 0.5,
        "window_recall": 0.5,
        "window_f1": 0.5,
        "roc_auc": 0.8,
        "pr_auc": 0.4,
        "threshold": 1.0,
    }


def test_verdict_names_the_baseline_when_validation_prefers_it() -> None:
    rows = [
        row("window_spread", "smd", "time_based", 0.78),
        row("lstm_autoencoder[normal_val_loss]", "smd", "time_based", 0.63),
    ]

    text = "\n".join(verdict_lines(rows))

    assert "validation prefers `window_spread`" in text
    assert "-0.150" in text


def test_verdict_names_the_network_when_validation_prefers_it() -> None:
    rows = [
        row("window_spread", "smd", "time_based", 0.60),
        row("lstm_autoencoder[normal_val_loss]", "smd", "time_based", 0.75),
    ]

    assert "validation prefers the autoencoder" in "\n".join(verdict_lines(rows))


def test_the_winner_is_chosen_on_validation_not_on_test() -> None:
    """A detector that looks best on test but worse on validation is not called the winner."""
    rows = [
        row("window_spread", "smd", "time_based", 0.90, val_curve=0.40),
        row("ewma_residual", "smd", "time_based", 0.70, val_curve=0.60),
        row("lstm_autoencoder[normal_val_loss]", "smd", "time_based", 0.60, val_curve=0.50),
    ]

    text = "\n".join(verdict_lines(rows))

    assert "best without a network is `ewma_residual`" in text
    assert "validation prefers `ewma_residual`" in text


def test_a_tie_is_reported_as_one() -> None:
    rows = [
        row("window_spread", "synthetic", "time_based", 1.0, delay=19.0),
        row("lstm_autoencoder[normal_val_loss]", "synthetic", "time_based", 1.0, delay=16.0),
    ]

    assert "validation cannot separate them" in "\n".join(verdict_lines(rows))


def test_a_detector_living_on_standing_alarms_is_named() -> None:
    rows = [
        row("window_spread", "smd", "time_based", 0.78, pre_alarmed=0.27, worst_duty=1.0),
        row("lstm_autoencoder[normal_val_loss]", "smd", "time_based", 0.63),
    ]

    text = "\n".join(verdict_lines(rows))

    assert "### Standing alarms" in text
    assert "`window_spread`: 27% of detections pre-alarmed" in text
    assert "lstm_autoencoder" not in text.split("### Standing alarms")[1]


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

    assert "validation prefers" not in text


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


def test_detectors_are_listed_by_curve_area() -> None:
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
