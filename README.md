# DrDoom

Autonomous incident response for production services: detect an anomaly, diagnose the
root cause against real operational documentation, propose a remediation, hold it at a
human approval gate, and write the postmortem.

> **Status.** Every layer in the table below is built and tested. The running service
> scores windows posted to it in the synthetic four-metric schema; it does not yet read
> from a live metrics source.

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
| Agents and model layer | Done |
| Agent orchestration | Done |
| Observability | Done |
| Approval gate and audit | Done |
| API and dashboard | Done |

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

### Agents

Four agents: triage decides whether anything is wrong and what kind, diagnosis explains it
from retrieved documentation, remediation proposes a risk-rated fix, and reporting writes
the postmortem once a decision exists.

Two properties matter more than the prompts.

**Whether an action needs a human is decided by the type, not the model.** On
`RemediationPlan`, `requires_approval` is a computed field derived from `risk_level`. The
model is never asked for it, and a value supplied anyway is ignored — a system whose
safety argument is "a human approves risky actions" cannot let the supervised thing
decide what counts as risky.

**Nor does the plan's author get the last word on its risk.** The rating the gate uses is
the most cautious of three: a policy floor fixed in code for each catalogue action (a node
drain is always high, only scaling out can be low), an independent assessment from a
separate call that is shown the incident, the plan and the exact command but not the
author's rating, and the author's own rating. None of the three can lower another. An
assessor that cannot answer counts as high. The gate and the audit log show all three,
and the approval hash covers the final rating. Set `DRDOOM_RISK_PROVIDER` or
`DRDOOM_RISK_MODEL` to have a different model do the reviewing.

**Every structured response is validated, and one repair is allowed.** A malformed plan
goes back to the model with the validation error attached, once. Not a loop: a model that
cannot satisfy a schema on the second attempt rarely does on the fifth, and an unbounded
repair turns a bad response into a bill. If the provider is unreachable, or its answer
still fails after the repair, the agents degrade to the retrieved documentation and say
which of the two happened, rather than inventing a summary. A plan written without a
usable answer proposes nothing, is rated high risk so it still stops at the gate, and
names no action the executor can run, so approving it changes nothing. Without a provider
key the service still starts in that degraded form, and `/health` says so. A run that
stops at an error is reported as `failed`, never as `no_incident`.

**What runs is a field, not a sentence.** `immediate_action` is prose for the human. The
executor reads only `action`, which can name one of five catalogue entries or nothing, and
the approval gate shows the exact command it would render. The prose is never searched for
keywords: a substring match once read "never roll back" as a rollback.

**A window is checked before anything is spent on it.** Missing or infinite values, a
length or metric order the threshold was not calibrated for, and oversized bodies are
refused with 422 before the graph starts. Unchecked, a single NaN compared false against the
threshold and was reported as an incident.

### Orchestration

One pipeline definition, and it suspends.

An investigation pauses to ask a human a question, and the answer may not arrive for
hours. A request handler cannot block on that. The graph therefore calls `interrupt()`
before the remediation gate; the checkpointer writes the suspended state to SQLite, and a
later call resumes the same `thread_id` from where it stopped.

Nothing is held in process memory between those two calls, which is the property that
makes this a graph rather than four function calls in sequence. The test that proves it
starts an investigation in one interpreter, exits, and finishes it in a second one that
shares nothing but the database file.

```
triage ─┬─ no incident ─────────────────────────────► end
        └─ incident ─► diagnose ─► remediate ─► assess_risk ─► approval ─► report ─► end
                                                                  │
                                                         suspends here when the
                                                         final risk is medium or high
```

Routing after triage is conditional, so a quiet system costs nothing: no retrieval, no
model call, no tokens.

### API and dashboard

```bash
DRDOOM_API_KEYS="aditya:choose-a-key" uv run uvicorn drdoom.api.main:app
```

`POST /investigate` starts a run and returns where it stopped. `POST /investigate/stream`
delivers the same run a stage at a time over server-sent events, so the dashboard fills in
progressively instead of blocking on one long request.
`POST /incidents/{id}/approve` resumes a suspended one.

The service retrieves with BM25 and the MiniLM encoder fused, the configuration the
evaluation suite scores. The first start embeds the document corpus, which takes a few
minutes on a CPU; the matrix is saved beside the corpus and every later start reuses it.

**Approving requires a credential**; reading does not. The key maps to a named principal,
because the audit log has to record *who* decided and "someone with a valid key" is not an
answer a review accepts. An unset key ring accepts nobody — a deployment that forgot to
configure credentials refuses approvals rather than accepting them from anyone.

**Approving twice is safe.** Networks retry, so a recorded decision is returned as it
stands rather than applied a second time. A later request with the opposite answer
changes nothing: the recorded decision is returned as it stands.

**No model output reaches the DOM as markup.** Plain fields go through `textContent`; the
postmortem is markdown, so it goes through DOMPurify. That chain matters here more than
usual: text originates from a model reading a document corpus, and the dashboard is
same-origin with the approval endpoint, so a poisoned document that talks the model into
emitting a script tag would otherwise be one step from a privileged action. A test asserts
every `innerHTML` assignment in the page has `DOMPurify.sanitize` on its right-hand side,
and the behaviour was checked in a browser against a payload carrying
`<script>`, `<img onerror>` and a `javascript:` link — all three are stripped and the text
displays inert.

### Evaluating the generated half

Three of the four agents are model calls. Until this existed, every published number
described the classical half, which is the easy half to measure.

The suite scores retrieval and the groundedness of generated diagnoses over 15 labelled
incident scenarios and 40 retrieval queries, and CI fails the build when a score drops
below its floor. It runs against **recorded** model responses, so it is free, offline and
identical on every run — which is what lets it gate a merge rather than being a script
somebody runs occasionally.

**Groundedness here is lexical support, not entailment.** Each sentence of a diagnosis is
scored by how much of its distinctive vocabulary appears in the passages retrieved for it.
A sentence can reuse the context's words and still be wrong, and a correct paraphrase
scores lower than it deserves. Read it as *how much of this answer is traceable to its
sources* — the question worth asking of a machine-written diagnosis — not as a truth score.

| Measure | Score | Floor |
|---|---:|---:|
| retrieval hit@5 | 0.850 | 0.75 |
| diagnosis retrieved the right document | 0.600 | 0.50 |
| groundedness | 0.594 | 0.50 |
| supported sentence fraction | 0.494 | 0.40 |
| structured output parsed | 1.000 | 1.00 |

Full report in [docs/eval-results.md](docs/eval-results.md). Floors sit *below* the
measured baseline on purpose — a floor set from an aspiration makes the suite permanently
red and therefore ignored.

The suite's most useful finding was about itself. An early version scored lexical
retrieval alone and reported numbers the deployed system would never produce. Symptom
descriptions are the case that separates them: *"services cannot resolve each other by
name"* shares no vocabulary with the page about DNS, so BM25 answered it with Vertical Pod
Autoscaling. Evaluating the shipped hybrid retriever instead moved retrieval hit@5 from
0.725 to 0.850 and the diagnosis retrieval rate from 0.333 to 0.600.

```bash
python -m drdoom.evals.run              # replay recorded responses
python -m drdoom.evals.run --record     # refresh them against a live provider
```

Risk ratings are scored separately, against sixteen hand-labelled plans, for how often a
plan ends up rated below what it deserves. The author's rating alone under-rates 8 of them;
adding the policy floor leaves 4, each of which only an assessor can catch. The row with the
assessor is measured once its answers are recorded
([docs/risk-results.md](docs/risk-results.md)).

```bash
python -m drdoom.evals.risk              # replay recorded assessor answers
python -m drdoom.evals.risk --record     # record them against a live provider
```

Token counts are reported on every API response, with an estimated cost where the model's
published rate has been checked. An unknown model reports tokens and no cost rather than a
guessed figure.

### Observability

Structured JSON logs, one object per line, with the incident id on **every** record — set
once when a run starts and carried through a context variable, so two concurrent
investigations never interleave into an unreadable stream.

```json
{"time":"2026-09-02T16:56:16","level":"INFO","logger":"drdoom.timing",
 "incident":"inc-demo-01","message":"stage complete","stage":"triage","duration_ms":0.4}
```

Every stage is timed, and `/metrics` reports p50/p95 per stage alongside request counts and
whether the audit chain still verifies — because "which stage was slow" is the first
question asked about a slow investigation, and it is unanswerable from a log that only
records what happened.

Deliberately not OpenTelemetry: a collector, exporter and backend is a lot of moving parts
for a service answering one question, and these records carry the same fields a span would.
The seam to swap in a real tracer is one function.

### Running it in a container

```bash
DRDOOM_API_KEYS="you:pick-a-key" GROQ_API_KEY=... docker compose up
```

Two stages: the builder installs dependencies and bakes in the document corpus, the
embedding model and the corpus embeddings, so a cold container answers immediately rather
than downloading half a gigabyte and embedding thousands of passages while someone waits.
Torch comes from the CPU-only index — the default wheels carry CUDA libraries worth several
gigabytes and nothing here uses a GPU. Runs as a non-root user, health-checks the
application rather than the port, and keeps `state/` on a volume so suspended
investigations survive a restart.

### Security

The system reads documents it does not control, hands them to a model, shows the result to
an engineer, and can act on that engineer's approval — a chain from untrusted input to
privileged action. [SECURITY.md](SECURITY.md) sets out the threat model, what is done about
each threat, and, more usefully, what is **not**: no rate limiting on `/investigate`, no
Subresource Integrity on the CDN scripts, static API keys with no expiry, and an audit
chain that is tamper-evident locally but not anchored anywhere an attacker could not reach.

### Model providers

| Mode | Requires | Cost |
|---|---|---|
| Tests and CI | nothing — a stub provider, no key, no network | free |
| Local and demo | `GROQ_API_KEY` | free tier |
| Optional | `ANTHROPIC_API_KEY` + `uv sync --extra anthropic` | paid |

Groq is the default because a demonstration nobody can afford to run is not a
demonstration. Anthropic is a swappable alternative, kept out of the default install so
nothing unused ships.

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
