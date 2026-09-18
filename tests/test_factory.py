"""What the dashboard's two demonstration buttons actually do.

The service is assembled from generated traffic so it runs without the real dataset, and
its threshold is measured on separately generated validation traffic rather than typed in.
These tests hold the assembled detector to what the buttons promise.
"""

import pytest

from drdoom.agents.triage import TriageAgent
from drdoom.api.factory import build_detector, demo_window


@pytest.fixture(scope="module")
def triage() -> TriageAgent:
    detector, threshold, feature_names = build_detector()
    return TriageAgent(detector, threshold, feature_names)


def test_the_disturbed_demo_window_raises_an_incident(triage) -> None:
    assert triage.run(demo_window(anomalous=True)).is_anomaly is True


def test_the_calm_demo_window_does_not(triage) -> None:
    assert triage.run(demo_window(anomalous=False)).is_anomaly is False


def test_the_threshold_is_identical_on_every_start() -> None:
    assert build_detector()[1] == build_detector()[1]
