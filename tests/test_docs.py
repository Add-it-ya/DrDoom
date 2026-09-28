"""Published results pages must be exactly what their data renders to."""

import json
from pathlib import Path

from drdoom.detect.compare import render_markdown

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_the_detection_page_is_rendered_from_its_json() -> None:
    """Hand edits to either file, or a stale copy of one, fail here."""
    data = json.loads((DOCS / "detection-results.json").read_text(encoding="utf-8"))
    published = (DOCS / "detection-results.md").read_text(encoding="utf-8")

    assert published == render_markdown(data["rows"], data["budget"])


def test_every_detection_row_carries_the_repaging_figures() -> None:
    data = json.loads((DOCS / "detection-results.json").read_text(encoding="utf-8"))

    for row in data["rows"]:
        assert {"pages_per_day", "alarm_duty", "fresh_detection_rate", "curve_auc"} <= set(row)
