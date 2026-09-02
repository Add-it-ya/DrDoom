"""Score the non-deterministic half of the system, and fail if it regresses.

Run with ``python -m drdoom.evals.run``.

Three of the four agents are model calls, and until this existed nothing measured any of
them. Every published number described the classical half, which is the easy half to
measure and the half least likely to surprise anyone.

The suite runs against recorded responses by default, so it is free, offline and
deterministic -- which is what lets continuous integration gate on it. Pass ``--record``
with a configured provider to refresh the snapshots after changing a prompt.

Thresholds are floors, not targets. They exist so a prompt edit that quietly stops
grounding answers in the retrieved text fails a build instead of shipping.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from drdoom.agents.diagnosis import DiagnosisAgent, format_passages
from drdoom.config import get_settings
from drdoom.evals.groundedness import score_text
from drdoom.llm.base import LLMProvider, LLMUnavailableError
from drdoom.llm.recording import RecordingProvider, ReplayProvider, SnapshotStore
from drdoom.rag import corpus
from drdoom.rag.evaluate import evaluate as evaluate_retrieval
from drdoom.rag.evaluate import load_queries
from drdoom.rag.index import BM25Index, Retriever
from drdoom.rag.ingest import chunk_all

logger = logging.getLogger(__name__)

# Floors, chosen a little below what the suite currently scores so ordinary variation
# does not fail a build but a real regression does.
THRESHOLDS = {
    "retrieval_hit_at_5": 0.60,
    "diagnosis_retrieval_hit": 0.60,
    "groundedness": 0.65,
    "supported_fraction": 0.60,
    "parse_success": 1.00,
}


def snapshot_dir() -> Path:
    return get_settings().project_root / "evals" / "snapshots"


def cases_path() -> Path:
    return get_settings().project_root / "evals" / "diagnosis_cases.json"


@dataclass
class CaseResult:
    case_id: str
    retrieved_expected: bool
    groundedness: float
    supported_fraction: float
    has_expected_terms: bool
    parsed: bool
    degraded: bool
    tokens: int
    unsupported: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "case": self.case_id,
            "retrieved_expected": self.retrieved_expected,
            "groundedness": round(self.groundedness, 4),
            "supported_fraction": round(self.supported_fraction, 4),
            "has_expected_terms": self.has_expected_terms,
            "parsed": self.parsed,
            "degraded": self.degraded,
            "tokens": self.tokens,
            "unsupported": self.unsupported,
        }


def load_cases() -> list[dict]:
    return json.loads(cases_path().read_text(encoding="utf-8"))["cases"]


def build_retriever() -> Retriever:
    if not corpus.is_downloaded():
        raise SystemExit(
            "the document corpus is missing; run: python -c "
            '"from drdoom.rag import corpus; corpus.download()"'
        )
    return BM25Index(chunk_all(corpus.load()))


def run_case(agent: DiagnosisAgent, case: dict) -> CaseResult:
    """Diagnose one scenario and score what came back against what was retrieved."""
    try:
        outcome = agent.run(case["symptoms"], case.get("root_cause"))
        parsed = True
    except LLMUnavailableError:
        raise
    except ValueError:
        logger.exception("case %s produced an unusable diagnosis", case["id"])
        return CaseResult(case["id"], False, 0.0, 0.0, False, False, False, 0)

    hits = agent.retrieve(case["symptoms"] + " " + (case.get("root_cause") or ""))
    context = format_passages(hits)
    answer = " ".join(
        [outcome.diagnosis.summary, outcome.diagnosis.likely_cause, outcome.diagnosis.next_action]
    )
    report = score_text(answer, context)

    retrieved = {citation.doc_id for citation in outcome.citations}
    expected = set(case.get("relevant", []))
    lowered = answer.lower()

    return CaseResult(
        case_id=case["id"],
        retrieved_expected=bool(retrieved & expected),
        groundedness=report.score,
        supported_fraction=report.supported_fraction,
        has_expected_terms=all(term in lowered for term in case.get("expect_terms", [])),
        parsed=parsed,
        degraded=outcome.degraded,
        tokens=outcome.tokens,
        unsupported=[item.sentence for item in report.unsupported][:2],
    )


def summarise(results: list[CaseResult], retrieval: dict) -> dict:
    total = max(len(results), 1)
    return {
        "cases": len(results),
        "retrieval_hit_at_5": retrieval["hit_at_5"],
        "retrieval_mrr": retrieval["mrr"],
        "retrieval_queries": retrieval["queries"],
        "diagnosis_retrieval_hit": sum(r.retrieved_expected for r in results) / total,
        "groundedness": sum(r.groundedness for r in results) / total,
        "supported_fraction": sum(r.supported_fraction for r in results) / total,
        "expected_terms_present": sum(r.has_expected_terms for r in results) / total,
        "parse_success": sum(r.parsed for r in results) / total,
        "degraded": sum(r.degraded for r in results),
        "total_tokens": sum(r.tokens for r in results),
    }


def check(summary: dict) -> list[str]:
    """Return the thresholds this run failed."""
    return [
        f"{name}: {summary[name]:.3f} below floor {floor:.2f}"
        for name, floor in THRESHOLDS.items()
        if name in summary and summary[name] < floor
    ]


def render_markdown(summary: dict, results: list[CaseResult], failures: list[str]) -> str:
    lines = [
        "# Evaluation of the generated half",
        "",
        "Generated by `python -m drdoom.evals.run`, against recorded model responses so the",
        "suite is free, offline and identical on every run. Continuous integration fails the",
        "build when a score drops below its floor.",
        "",
        "## What groundedness means here",
        "",
        "Each sentence of a diagnosis is scored by how much of its distinctive vocabulary",
        "appears in the passages retrieved for it. A sentence naming a mechanism or component",
        "absent from the context scores low.",
        "",
        "This is a proxy for entailment, not entailment. A sentence can reuse the context's",
        "words and still be wrong, and a correct paraphrase scores lower than it deserves.",
        "Read it as *how much of this answer is traceable to its sources*, which is the",
        "question worth asking of a machine-written diagnosis, not as a truth score.",
        "",
        "## Scores",
        "",
        "| Measure | Score | Floor |",
        "|---|---:|---:|",
    ]
    for name in (
        "retrieval_hit_at_5",
        "diagnosis_retrieval_hit",
        "groundedness",
        "supported_fraction",
        "expected_terms_present",
        "parse_success",
    ):
        floor = THRESHOLDS.get(name)
        lines.append(
            f"| {name.replace('_', ' ')} | {summary[name]:.3f} | "
            f"{f'{floor:.2f}' if floor else '-'} |"
        )

    lines += [
        "",
        f"{summary['cases']} diagnosis cases, {summary['retrieval_queries']} retrieval queries, "
        f"{summary['total_tokens']} tokens across the suite.",
        "",
        "## Per case",
        "",
        "| Case | Found the right document | Groundedness | Expected terms |",
        "|---|:--:|---:|:--:|",
    ]
    for result in results:
        lines.append(
            f"| {result.case_id} | {'yes' if result.retrieved_expected else 'no'} | "
            f"{result.groundedness:.3f} | {'yes' if result.has_expected_terms else 'no'} |"
        )

    if failures:
        lines += ["", "## Failures", ""] + [f"- {failure}" for failure in failures]
    lines.append("")
    return "\n".join(lines)


def _retrieval_only(retriever: Retriever, out: Path | None) -> int:
    """Score what can be scored without a model, and say plainly what was skipped.

    Exits zero: an unrecorded suite is a gap to fill, not a regression to block on.
    """
    metrics = evaluate_retrieval("bm25", retriever, load_queries())
    docs = out or get_settings().project_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    body = [
        "# Evaluation of the generated half",
        "",
        "Retrieval scored. The diagnosis half is **not recorded**, so groundedness is",
        "unmeasured. Refresh the snapshots with `python -m drdoom.evals.run --record`",
        "against a configured provider.",
        "",
        f"- retrieval hit@5: {metrics.hit_rate[5]:.3f} over {metrics.n_queries} queries",
        f"- retrieval MRR: {metrics.mrr:.3f}",
        "",
    ]
    (docs / "eval-results.md").write_text("\n".join(body), encoding="utf-8")

    logger.info("retrieval hit@5 %.3f over %d queries", metrics.hit_rate[5], metrics.n_queries)
    logger.warning("groundedness not measured: no recorded responses")
    return 0


def build_provider_for(record: bool, provider_name: str) -> LLMProvider:
    store = SnapshotStore(snapshot_dir())
    if not record:
        return ReplayProvider(store)

    from drdoom.llm.factory import build_provider

    return RecordingProvider(build_provider(provider_name), store, note="diagnosis eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score retrieval and diagnosis quality.")
    parser.add_argument(
        "--record", action="store_true", help="call a real provider and refresh snapshots"
    )
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    retriever = build_retriever()
    store = SnapshotStore(snapshot_dir())
    if not args.record and len(store) == 0:
        logger.warning(
            "no recorded responses in %s; the diagnosis half of this suite cannot run. "
            "Refresh it with: python -m drdoom.evals.run --record",
            snapshot_dir(),
        )
        return _retrieval_only(retriever, args.out)

    provider = build_provider_for(args.record, args.provider)
    agent = DiagnosisAgent(retriever, provider)

    retrieval_metrics = evaluate_retrieval("bm25", retriever, load_queries())
    retrieval = {
        "hit_at_5": retrieval_metrics.hit_rate[5],
        "mrr": retrieval_metrics.mrr,
        "queries": retrieval_metrics.n_queries,
    }

    results = [run_case(agent, case) for case in load_cases()]
    summary = summarise(results, retrieval)
    failures = check(summary)

    docs = args.out or get_settings().project_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "eval-results.json").write_text(
        json.dumps({"summary": summary, "cases": [r.as_dict() for r in results]}, indent=2),
        encoding="utf-8",
    )
    (docs / "eval-results.md").write_text(
        render_markdown(summary, results, failures), encoding="utf-8"
    )

    for name, value in summary.items():
        logger.info("%-26s %s", name, round(value, 4) if isinstance(value, float) else value)

    if failures:
        for failure in failures:
            logger.error("FAILED %s", failure)
        return 1
    logger.info("all evaluation floors met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
