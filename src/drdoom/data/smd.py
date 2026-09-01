"""Server Machine Dataset: download, parse and turn into annotated metric series.

The dataset ships five weeks of telemetry from 28 production machines, 38 metrics each.
Its own ``train`` split is anomaly-free and its ``test`` split carries point labels plus
an ``interpretation_label`` file naming which metric dimensions deviated during each
anomaly.

Two parsing decisions worth stating, both verified against the raw files:

* Interpretation ranges are zero-based and end-exclusive.
* Ranges do not always cover their label run exactly -- on some machines a labelled run
  extends past the interpretation range that describes it. Events are therefore derived
  from contiguous runs in ``test_label`` (the authoritative detection ground truth) and
  dimensions are attached from whichever interpretation ranges overlap the run.
"""

from __future__ import annotations

import logging
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from drdoom.config import get_settings
from drdoom.data.schema import AnomalyEvent, MetricSeries, event_id_track, find_label_runs

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://codeload.github.com/NetManAIOps/OmniAnomaly/tar.gz/refs/heads/master"
ARCHIVE_SUBDIR = "ServerMachineDataset"
SOURCE = "smd"
N_FEATURES = 38

MACHINES: tuple[str, ...] = tuple(
    f"machine-{group}-{index}"
    for group, count in ((1, 8), (2, 9), (3, 11))
    for index in range(1, count + 1)
)

FEATURE_NAMES: list[str] = [f"m{i:02d}" for i in range(N_FEATURES)]


def dataset_dir() -> Path:
    return get_settings().raw_data_dir / ARCHIVE_SUBDIR


def is_downloaded() -> bool:
    root = dataset_dir()
    return all((root / part).is_dir() for part in ("train", "test", "test_label")) and len(
        list((root / "train").glob("*.txt"))
    ) == len(MACHINES)


def download(force: bool = False) -> Path:
    """Stream the archive and extract only the dataset directory."""
    root = dataset_dir()
    if is_downloaded() and not force:
        logger.info("dataset already present at %s", root)
        return root

    destination = root.parent
    destination.mkdir(parents=True, exist_ok=True)
    logger.info("downloading server machine dataset to %s", destination)

    written = 0
    with (
        urllib.request.urlopen(ARCHIVE_URL) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        for member in archive:
            relative = _safe_member_path(member.name)
            if relative is None or not member.isfile():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with target.open("wb") as handle:
                handle.write(extracted.read())
            written += 1

    logger.info("extracted %d files", written)
    return root


def _safe_member_path(name: str) -> Path | None:
    """Map an archive member onto a relative path under the dataset directory.

    Returns ``None`` for anything outside the dataset directory, and rejects absolute
    paths or parent traversal so a malformed archive cannot write outside the target.
    """
    candidate = Path(name)
    parts = candidate.parts
    if candidate.is_absolute() or ".." in parts:
        return None
    if ARCHIVE_SUBDIR not in parts:
        return None
    return Path(*parts[parts.index(ARCHIVE_SUBDIR) :])


def _read_matrix(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, header=None, dtype=np.float32)
    return np.ascontiguousarray(frame.to_numpy(dtype=np.float32))


def _read_point_labels(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, header=None, dtype=np.int8)
    return frame.to_numpy(dtype=np.int8).ravel()


def parse_interpretation(path: Path) -> list[tuple[int, int, tuple[int, ...]]]:
    """Parse ``start-end:dim,dim`` lines into zero-based, end-exclusive spans."""
    if not path.is_file():
        return []
    spans: list[tuple[int, int, tuple[int, ...]]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        span, dims = line.split(":", 1)
        start_text, _, end_text = span.partition("-")
        dimensions = tuple(sorted(int(d) for d in dims.split(",") if d.strip()))
        spans.append((int(start_text), int(end_text), dimensions))
    return spans


def _events_for_machine(
    machine: str,
    point_labels: np.ndarray,
    spans: list[tuple[int, int, tuple[int, ...]]],
    next_event_id: int,
) -> list[AnomalyEvent]:
    events: list[AnomalyEvent] = []
    for offset, (start, end) in enumerate(find_label_runs(point_labels)):
        dims: set[int] = set()
        for span_start, span_end, span_dims in spans:
            if span_start < end and start < span_end:
                dims.update(span_dims)
        events.append(
            AnomalyEvent(
                event_id=next_event_id + offset,
                source=SOURCE,
                series_id=machine,
                start=start,
                end=end,
                dims=tuple(sorted(dims)),
            )
        )
    return events


def load_machine(machine: str, next_event_id: int = 0) -> tuple[MetricSeries, MetricSeries]:
    """Return the anomaly-free train series and the labelled test series for a machine."""
    root = dataset_dir()
    train_values = _read_matrix(root / "train" / f"{machine}.txt")
    test_values = _read_matrix(root / "test" / f"{machine}.txt")
    point_labels = _read_point_labels(root / "test_label" / f"{machine}.txt")

    if len(point_labels) != len(test_values):
        raise ValueError(f"{machine}: {len(point_labels)} labels for {len(test_values)} test rows")

    spans = parse_interpretation(root / "interpretation_label" / f"{machine}.txt")
    events = _events_for_machine(machine, point_labels, spans, next_event_id)

    train = MetricSeries(
        source=SOURCE,
        series_id=machine,
        values=train_values,
        point_labels=np.zeros(len(train_values), dtype=np.int8),
        event_ids=event_id_track(len(train_values), []),
        events=[],
        feature_names=list(FEATURE_NAMES),
    )
    test = MetricSeries(
        source=SOURCE,
        series_id=machine,
        values=test_values,
        point_labels=point_labels,
        event_ids=event_id_track(len(test_values), events),
        events=events,
        feature_names=list(FEATURE_NAMES),
    )
    return train, test


def load_all(machines: tuple[str, ...] = MACHINES) -> dict[str, tuple[MetricSeries, MetricSeries]]:
    """Load every machine, assigning globally unique event ids in machine order."""
    series: dict[str, tuple[MetricSeries, MetricSeries]] = {}
    next_event_id = 0
    for machine in machines:
        train, test = load_machine(machine, next_event_id=next_event_id)
        next_event_id += len(test.events)
        series[machine] = (train, test)
    return series
