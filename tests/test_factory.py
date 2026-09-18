"""What the dashboard's two demonstration buttons actually do.

The service is assembled from generated traffic so it runs without the real dataset, and
its threshold is measured on separately generated validation traffic rather than typed in.
These tests hold the assembled detector to what the buttons promise.
"""

import pytest

from drdoom.agents.triage import TriageAgent
from drdoom.api.factory import WINDOW, build_detector, demo_window
from drdoom.data import synthetic


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


@pytest.mark.parametrize("anomalous", [True, False])
def test_the_demo_windows_fit_the_service_exactly(anomalous: bool) -> None:
    """The service refuses any other shape, so the dashboard's buttons must produce this one."""
    assert demo_window(anomalous=anomalous).shape == (WINDOW, len(synthetic.FEATURE_NAMES))
