"""Pages that say they are generated have to be generated, figures and all."""

from pathlib import Path

import pytest

from drdoom.classify import card as classifier_card
from drdoom.data import card as dataset_card

DOCS = Path(__file__).resolve().parents[1] / "docs"


def manifest(source: str, strategy: str) -> dict:
    split = {
        "series": 28,
        "timesteps": 708405,
        "events": 197,
        "windows": 35271,
        "anomaly_window_rate": 0.094,
        "median_event_length": 21,
    }
    return {
        "source": source,
        "strategy": strategy,
        "splits": {"train": split, "val": split, "test": split},
    }


def test_the_dataset_table_is_rendered_from_the_manifest() -> None:
    page = dataset_card.render([manifest("smd", "time_based")])

    assert "| smd | time_based | test | 28 | 708,405 | 197 | 35,271 | 9.4% | 21 |" in page
    assert page.startswith("# Dataset composition")


def test_every_manifest_on_disk_is_listed_in_order(tmp_path) -> None:
    import json

    for source, strategy in [("synthetic", "time_based"), ("smd", "held_out_series")]:
        folder = tmp_path / source / strategy
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps(manifest(source, strategy)))

    found = dataset_card.load_manifests(tmp_path)

    assert [(m["source"], m["strategy"]) for m in found] == [
        ("smd", "held_out_series"),
        ("synthetic", "time_based"),
    ]


def test_no_manifests_leaves_the_existing_page_alone(tmp_path) -> None:
    assert dataset_card.write_page([], docs_root=tmp_path) is None
    assert not (tmp_path / "dataset.md").exists()


def test_the_published_dataset_page_matches_what_the_build_would_write() -> None:
    manifests = dataset_card.load_manifests()
    if len(manifests) < 4:
        pytest.skip("the datasets are not built here")
    published = (DOCS / "dataset.md").read_text(encoding="utf-8")

    assert published == dataset_card.render(manifests)


def test_the_incident_count_is_counted_not_typed() -> None:
    summaries = [
        {"source": "smd", "archetype_support": {"narrow": 181, "broad": 109, "other": 37}},
        {"source": "synthetic"},
    ]

    assert classifier_card._incident_total(summaries) == "327 incidents"
    assert classifier_card._incident_total([{"source": "synthetic"}]) == "a few hundred incidents"


def test_the_published_classifier_card_matches_its_summary() -> None:
    import json

    summaries = json.loads((DOCS / "classifier-summary.json").read_text(encoding="utf-8"))
    published = (DOCS / "classifier-card.md").read_text(encoding="utf-8")

    assert published == classifier_card.render(summaries)
