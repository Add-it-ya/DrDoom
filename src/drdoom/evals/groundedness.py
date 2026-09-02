"""Does the diagnosis say anything the retrieved documents do not support.

**What this measures, and what it does not.** Groundedness here is lexical support: each
sentence of a diagnosis is compared against the passages that were retrieved for it, and
scored by how much of its distinctive vocabulary appears there. A sentence built entirely
from words present in the context scores high; a sentence naming a mechanism, a number or
a component that appears nowhere in the context scores low.

That is a proxy for entailment, not entailment. A sentence can reuse the context's
vocabulary and still assert something false, and a correct paraphrase using different
words will score lower than it deserves. Reported numbers should be read as "how much of
this answer is traceable to the sources", which is the question an on-call engineer asks
of a machine-written diagnosis, rather than as a truth score.

It is deterministic and free, which is the reason it and not a model sits in continuous
integration. An optional judge with a written rubric is available for a deeper look, and
its verdicts are snapshotted like any other provider call.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

WORD = re.compile(r"[a-z0-9]{3,}")
SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Words that carry no evidential weight, so their presence in the context proves nothing.
FILLER: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "have",
        "has",
        "had",
        "are",
        "was",
        "were",
        "been",
        "being",
        "will",
        "would",
        "should",
        "could",
        "can",
        "may",
        "might",
        "must",
        "not",
        "but",
        "its",
        "their",
        "there",
        "here",
        "then",
        "than",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "too",
        "very",
        "just",
        "also",
        "into",
        "over",
        "under",
        "about",
        "after",
        "before",
        "because",
        "while",
        "during",
        "against",
        "between",
        "through",
        "above",
        "below",
        "down",
        "out",
        "off",
        "again",
        "further",
        "once",
        "you",
        "your",
        "they",
        "them",
        "our",
        "ours",
        "use",
        "used",
        "using",
        "make",
        "makes",
        "made",
        "need",
        "needs",
        "likely",
        "probable",
        "probably",
        "appears",
        "seems",
        "suggests",
        "indicates",
        "consider",
    }
)

SUPPORTED_AT = 0.6
MIN_CONTENT_WORDS = 3


def content_words(text: str) -> set[str]:
    """The words in a sentence that could actually be evidence."""
    return {word for word in WORD.findall(text.lower()) if word not in FILLER}


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE.split(text.strip()) if part.strip()]


@dataclass(frozen=True)
class SentenceScore:
    sentence: str
    support: float
    unsupported_terms: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return self.support >= SUPPORTED_AT


@dataclass(frozen=True)
class GroundednessReport:
    """How much of an answer traces back to its sources."""

    score: float
    sentences: list[SentenceScore] = field(default_factory=list)

    @property
    def supported_fraction(self) -> float:
        if not self.sentences:
            return 1.0
        return sum(1 for item in self.sentences if item.supported) / len(self.sentences)

    @property
    def unsupported(self) -> list[SentenceScore]:
        return [item for item in self.sentences if not item.supported]

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "supported_fraction": round(self.supported_fraction, 4),
            "sentences": len(self.sentences),
            "unsupported": [
                {"sentence": item.sentence, "support": round(item.support, 3)}
                for item in self.unsupported
            ],
        }


def score_text(text: str, context: str) -> GroundednessReport:
    """Score every sentence of ``text`` against the vocabulary of ``context``."""
    available = content_words(context)
    scores: list[SentenceScore] = []

    for sentence in split_sentences(text):
        words = content_words(sentence)
        if len(words) < MIN_CONTENT_WORDS:
            # Too short to carry a claim; scoring it would only add noise.
            continue
        missing = words - available
        support = 1.0 - (len(missing) / len(words))
        scores.append(
            SentenceScore(
                sentence=sentence,
                support=support,
                unsupported_terms=tuple(sorted(missing)[:8]),
            )
        )

    overall = sum(item.support for item in scores) / len(scores) if scores else 1.0
    return GroundednessReport(score=overall, sentences=scores)


JUDGE_RUBRIC = """You are auditing a machine-written incident diagnosis for groundedness.

Groundedness means every factual claim in the diagnosis is supported by the excerpts
provided. Judge only support, not correctness: a claim that is true in the world but
absent from the excerpts is ungrounded.

Score from 0 to 1:
  1.0  every claim is stated or directly implied by the excerpts
  0.7  the main claim is supported; a minor detail is not
  0.4  the main claim is only loosely related to the excerpts
  0.0  the diagnosis asserts things the excerpts do not mention at all

Return strict JSON: {"score": <number>, "unsupported": [<claim>, ...], "reason": "<one sentence>"}
"""


def judge_prompt(diagnosis: str, context: str) -> str:
    return f"EXCERPTS:\n{context}\n\nDIAGNOSIS:\n{diagnosis}\n\nScore its groundedness."
