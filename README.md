# DrDoom

Autonomous incident response for production services: detect an anomaly, diagnose the
root cause against real operational documentation, propose a remediation, hold it at a
human approval gate, and write the postmortem.

> ⚠️ **Early development.** The foundations are in place; the detection, retrieval and
> agent layers are being built. Results tables below are filled in as each lands.

## Why this exists

Most "AI for operations" projects are a chat window over log search. DrDoom is a closed
loop with a real gate in the middle: a statistical detection layer decides *whether*
something is wrong, a classifier decides *what kind* of wrong, retrieval grounds the
explanation in documentation that actually exists, and no remediation runs until a human
authorises it — with the decision written to an append-only audit log.

## Status

| Component | State |
|---|---|
| Project foundations | Done |
| Data pipeline | Done |
| Anomaly detection | Done |
| Root-cause classification | Done |
| Retrieval | Done |
| Agent orchestration | Not started |
| Approval gate and audit | Not started |
| API and dashboard | Not started |

## Data

Real telemetry from the Server Machine Dataset (28 machines, 38 metrics) as the primary
benchmark, with a synthetic generator alongside it for controlled experiments where a
known root cause and a severity dial are needed. See
[docs/dataset.md](docs/dataset.md) for the full composition.

```bash
python -m drdoom.data.build
```

## Results

Every metric is reported at incident level with 95% bootstrap confidence intervals,
alongside the simple baselines it is measured against. Full tables in
[docs/detection-results.md](docs/detection-results.md).

### Detection

Baselines were built before the network so the comparison could actually be made, and
thresholds are chosen on validation against a false alarm budget of one page per
series-day, then applied unchanged to test.

| Dataset | Best detector | Detection rate | Minutes to detect | Alarms/day |
|---|---|---:|---:|---:|
| Real, unseen machines | `ewma_residual` | 0.752 | 8.5 | 1.65 |
| Real, future incidents | `window_spread` | 0.782 | 8.0 | 0.92 |
| Synthetic, unseen services | `lstm_autoencoder` | 1.000 | 15.0 | 1.16 |

The autoencoder wins on synthetic data and **loses to a one-line statistic on real
data**, where it trails the best baseline by 0.15 to 0.20 detection rate while raising
more false alarms. That result is published rather than buried: it is the reason the
baselines exist, and the reason the shipped default on real telemetry is not the
neural network.

Results here are not point-adjusted. Much of the published work on this benchmark
credits a whole anomaly segment as detected when any single point inside it fires,
which inflates F1 and is not comparable to these numbers.

```bash
python -m drdoom.detect.compare
```

### Root cause

Once a window is flagged, a gradient-boosted classifier over window summary statistics
names the cause. Cross-validation folds are grouped by incident so overlapping slices of
one outage never straddle a fold, and verdicts are pooled per incident by majority vote.
Full card in [docs/classifier-card.md](docs/classifier-card.md).

| Dataset | Classes | Test incidents | Event macro F1 | 95% CI |
|---|---|---:|---:|---|
| Real telemetry | 2 signature archetypes | 59 | 0.827 | [0.726, 0.915] |
| Synthetic | 4 causal classes | 300 | 0.997 | [0.989, 1.000] |

The two are never pooled, because they are not the same task. The synthetic generator
picks a cause and writes its signature into the metrics, so 0.997 measures the generator's
separability more than the model's skill. The real dataset labels which of its 38
dimensions deviated but publishes no mapping from dimension to meaning, so its classes are
signature archetypes — narrow against broad — which makes that task subsystem attribution
rather than causal analysis. The weaker claim is the true one.

```bash
python -m drdoom.classify.train
```

### Retrieval

430 documents of Kubernetes and Prometheus operational documentation, 5671 chunks, with
no filter narrowing the search to a document the classifier already picked. Scored on 40
hand-authored operational questions in [`evals/`](evals/retrieval_queries.json). Full
ablation in [docs/retrieval-results.md](docs/retrieval-results.md).

| Configuration | Hit@5 | Recall@5 | MRR |
|---|---:|---:|---:|
| BM25 only | 0.725 | 0.675 | 0.539 |
| Dense only (MiniLM) | 0.825 | 0.787 | 0.630 |
| Hybrid (BM25 + MiniLM) | 0.850 | 0.812 | 0.617 |
| Hybrid + cross-encoder rerank | **0.875** | **0.838** | **0.699** |

Here, unlike detection and classification, every component pays for itself. The reranker
is the clearest case: it adds little to hit rate but moves MRR from 0.617 to 0.699, which
is what reranking is for — it does not find more, it orders better, and position decides
what fits in the model's context.

The dense index is plain numpy. At a few thousand chunks an exact dot product is well
under a millisecond; a vector service earns its place when the corpus outgrows memory or
needs concurrent writers, and adding one earlier would be the unjustified machinery this
project exists to avoid.

```bash
python -m drdoom.rag.evaluate
```

## Getting started

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Add-it-ya/DrDoom.git
cd DrDoom
uv sync
cp .env.example .env
```

Run the checks:

```bash
uv run ruff check .
uv run pytest
```

## Licence

MIT. See [LICENSE](LICENSE).
