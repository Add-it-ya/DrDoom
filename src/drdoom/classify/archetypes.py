"""Derive incident archetypes for the real dataset from its interpretation labels.

The Server Machine Dataset labels *when* an anomaly happened and *which of the 38
dimensions deviated*, but it does not say what any dimension measures. No column
mapping is published with the data or the paper. Naming groups of dimensions "cpu" or
"memory" would therefore be invention dressed as ground truth.

What the interpretation labels do support is grouping incidents by the *shape* of their
deviation signature. Clustering the signatures finds two well-populated groups: one
confined to a small set of correlated dimensions, one spread across many. Those are
described here as ``narrow_signature`` and ``broad_signature`` -- claims about extent,
which the data supports, rather than claims about cause, which it does not.

The resulting task is subsystem-signature attribution, not causal root-cause analysis.
The synthetic corpus carries true causal labels and is where the causal claim is tested;
these two are reported separately and never pooled.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from drdoom.data.schema import AnomalyEvent

OTHER_LABEL = "other"
DEFAULT_CLUSTERS = 6
DEFAULT_MIN_SUPPORT = 20


@dataclass(frozen=True)
class Archetypes:
    """A fitted assignment from event signature to archetype name."""

    labels: dict[int, str]
    profiles: dict[str, tuple[int, ...]]
    support: dict[str, int]

    def label_for(self, event: AnomalyEvent) -> str:
        return self.labels.get(event.event_id, OTHER_LABEL)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.support))


def signature_matrix(events: list[AnomalyEvent], n_features: int) -> np.ndarray:
    """One row per event, one column per dimension, marking the deviating dimensions.

    Interpretation dimensions are one-based in the source files.
    """
    matrix = np.zeros((len(events), n_features), dtype=np.float32)
    for row, event in enumerate(events):
        for dimension in event.dims:
            if 1 <= dimension <= n_features:
                matrix[row, dimension - 1] = 1.0
    return matrix


def _breadth_name(median_dims: float, n_features: int) -> str:
    """Name a cluster by how much of the metric space it touches."""
    return "narrow_signature" if median_dims <= max(3, n_features // 8) else "broad_signature"


def derive(
    events: list[AnomalyEvent],
    n_features: int,
    n_clusters: int = DEFAULT_CLUSTERS,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> Archetypes:
    """Cluster event signatures and keep only the groups with enough events to learn.

    Clusters below ``min_support`` collapse into ``other``. A class carrying five events
    across three splits teaches a model nothing and makes its reported score meaningless,
    which is the failure this threshold exists to prevent.
    """
    labelled = [event for event in events if event.dims]
    if len(labelled) < n_clusters:
        return Archetypes(labels={}, profiles={}, support={OTHER_LABEL: len(events)})

    matrix = signature_matrix(labelled, n_features)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    # Normalising makes the comparison about which dimensions co-occur rather than how
    # many of them fired, so a wide and a narrow event with the same shape stay together.
    assignments = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    ).fit_predict(matrix / norms)

    counts = Counter(assignments.tolist())
    labels: dict[int, str] = {}
    profiles: dict[str, tuple[int, ...]] = {}
    used: Counter[str] = Counter()

    for cluster, size in counts.most_common():
        members = np.flatnonzero(assignments == cluster)
        if size < min_support:
            continue
        median_dims = float(np.median(matrix[members].sum(axis=1)))
        name = _breadth_name(median_dims, n_features)
        used[name] += 1
        if used[name] > 1:
            name = f"{name}_{used[name]}"
        share = matrix[members].mean(axis=0)
        profiles[name] = tuple(int(d + 1) for d in np.argsort(-share)[:8] if share[d] > 0.25)
        for row in members:
            labels[labelled[row].event_id] = name

    support = Counter(labels.values())
    support[OTHER_LABEL] = len(events) - sum(support.values())
    return Archetypes(labels=labels, profiles=profiles, support=dict(support))
