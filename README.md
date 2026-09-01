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
| Anomaly detection | Not started |
| Root-cause classification | Not started |
| Retrieval | Not started |
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

Populated as each layer lands. Every metric is reported at incident level with
confidence intervals, alongside the simple baselines it is measured against.

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
