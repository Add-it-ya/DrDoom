"""Parsing rules for the Server Machine Dataset.

The interpretation ranges are zero-based and end-exclusive, and they do not always
cover their label run exactly, so events come from the label runs rather than from the
interpretation file. These tests pin both behaviours.
"""

from pathlib import Path

import numpy as np
import pytest

from drdoom.data import smd


def test_machine_list_covers_all_twenty_eight() -> None:
    assert len(smd.MACHINES) == 28
    assert len(set(smd.MACHINES)) == 28
    assert smd.MACHINES[0] == "machine-1-1"
    assert smd.MACHINES[-1] == "machine-3-11"


def test_parse_interpretation_reads_spans_and_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "machine-1-1.txt"
    path.write_text("15849-16368:1,9,10\n16963-17517:3,2\n\n")

    assert smd.parse_interpretation(path) == [
        (15849, 16368, (1, 9, 10)),
        (16963, 17517, (2, 3)),
    ]


def test_parse_interpretation_of_missing_file_is_empty(tmp_path: Path) -> None:
    assert smd.parse_interpretation(tmp_path / "absent.txt") == []


def test_events_come_from_label_runs_not_interpretation_spans() -> None:
    labels = np.zeros(30, dtype=np.int8)
    labels[10:20] = 1
    spans = [(10, 16, (2, 5))]

    events = smd._events_for_machine("machine-1-1", labels, spans, next_event_id=0)

    assert len(events) == 1
    assert (events[0].start, events[0].end) == (10, 20)
    assert events[0].dims == (2, 5)


def test_overlapping_interpretation_spans_merge_their_dimensions() -> None:
    labels = np.zeros(30, dtype=np.int8)
    labels[10:20] = 1
    spans = [(10, 14, (1,)), (14, 20, (7, 1)), (25, 28, (9,))]

    events = smd._events_for_machine("m", labels, spans, next_event_id=5)

    assert events[0].event_id == 5
    assert events[0].dims == (1, 7)


def test_event_ids_continue_from_the_given_offset() -> None:
    labels = np.zeros(20, dtype=np.int8)
    labels[2:4] = 1
    labels[10:12] = 1

    events = smd._events_for_machine("m", labels, [], next_event_id=100)

    assert [event.event_id for event in events] == [100, 101]


@pytest.mark.parametrize(
    "name",
    [
        "OmniAnomaly-master/README.md",
        "OmniAnomaly-master/data/other.txt",
        "../ServerMachineDataset/evil.txt",
    ],
)
def test_archive_members_outside_the_dataset_are_ignored(name: str) -> None:
    assert smd._safe_member_path(name) is None


def test_archive_member_paths_are_made_relative_to_the_dataset() -> None:
    result = smd._safe_member_path("OmniAnomaly-master/ServerMachineDataset/train/machine-1-1.txt")

    assert result == Path("ServerMachineDataset/train/machine-1-1.txt")


@pytest.mark.requires_dataset
def test_loaded_machine_agrees_with_its_labels() -> None:
    if not smd.is_downloaded():
        pytest.skip("dataset not downloaded")

    train, test = smd.load_machine("machine-1-1")

    assert train.n_features == test.n_features == smd.N_FEATURES
    assert int(train.point_labels.sum()) == 0
    assert int((test.event_ids != -1).sum()) == int(test.point_labels.sum())
    assert sum(event.length for event in test.events) == int(test.point_labels.sum())
