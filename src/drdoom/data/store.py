"""Persist and reload split metric series.

Series are stored as one concatenated value matrix plus offsets rather than as separate
arrays, so a whole split loads in a single read. Events go to a JSON sidecar because they
are small, ragged, and worth being able to read without numpy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from drdoom.data.schema import AnomalyEvent, MetricSeries, event_id_track


def save_series(path: Path, series: list[MetricSeries]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not series:
        raise ValueError("nothing to save")

    lengths = [item.n_timesteps for item in series]
    offsets = np.zeros(len(series) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    np.savez(
        path.with_suffix(".npz"),
        values=np.concatenate([item.values for item in series], axis=0),
        point_labels=np.concatenate([item.point_labels for item in series]),
        event_ids=np.concatenate([item.event_ids for item in series]),
        offsets=offsets,
    )

    sidecar = {
        "source": series[0].source,
        "feature_names": series[0].feature_names,
        "series": [
            {
                "series_id": item.series_id,
                "events": [
                    {
                        "event_id": event.event_id,
                        "start": event.start,
                        "end": event.end,
                        "dims": list(event.dims),
                        "label": event.label,
                    }
                    for event in item.events
                ],
            }
            for item in series
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")


def load_series(path: Path) -> list[MetricSeries]:
    payload = np.load(path.with_suffix(".npz"))
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

    offsets = payload["offsets"]
    source = sidecar["source"]
    feature_names = sidecar["feature_names"]

    series: list[MetricSeries] = []
    for position, entry in enumerate(sidecar["series"]):
        begin, end = int(offsets[position]), int(offsets[position + 1])
        events = [
            AnomalyEvent(
                event_id=record["event_id"],
                source=source,
                series_id=entry["series_id"],
                start=record["start"],
                end=record["end"],
                dims=tuple(record["dims"]),
                label=record["label"],
            )
            for record in entry["events"]
        ]
        values = payload["values"][begin:end]
        series.append(
            MetricSeries(
                source=source,
                series_id=entry["series_id"],
                values=values,
                point_labels=payload["point_labels"][begin:end],
                event_ids=event_id_track(len(values), events),
                events=events,
                feature_names=list(feature_names),
            )
        )
    return series
