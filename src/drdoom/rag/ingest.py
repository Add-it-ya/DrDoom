"""Split documents into retrievable chunks.

Chunks follow the document's own headings first, because a section is a unit an author
already decided was coherent. Sections longer than the size limit are then windowed with
overlap, so a passage that straddles a cut is still whole in one of the pieces.

Every chunk carries the heading path it came from and the provenance of its document, so
a retrieved passage can be shown with a citation rather than as an anonymous fragment.
Chunk ids are derived from the document id and character offset, which makes them stable
across rebuilds as long as the document has not changed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from drdoom.rag.corpus import Document

HEADING_LINE = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)

TARGET_CHARS = 1200
OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 120


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage, with enough context to cite and to read alone."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    heading: str
    text: str
    url: str
    licence: str
    offset: int

    @property
    def citation(self) -> str:
        return f"{self.title} - {self.heading}" if self.heading else self.title

    @property
    def search_text(self) -> str:
        """Text used for indexing, with the headings prepended for context."""
        prefix = f"{self.title}. {self.heading}. " if self.heading else f"{self.title}. "
        return prefix + self.text


def _sections(text: str) -> list[tuple[str, str, int]]:
    """Split on markdown headings into ``(heading, body, offset)`` triples."""
    matches = list(HEADING_LINE.finditer(text))
    if not matches:
        return [("", text, 0)]

    sections: list[tuple[str, str, int]] = []
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()], 0))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        sections.append((match.group(2).strip(), body, match.end()))
    return sections


def _windows(body: str, offset: int) -> list[tuple[str, int]]:
    """Cut an over-long section into overlapping pieces."""
    body = body.strip()
    if len(body) <= TARGET_CHARS:
        return [(body, offset)]

    pieces: list[tuple[str, int]] = []
    step = TARGET_CHARS - OVERLAP_CHARS
    for start in range(0, len(body), step):
        piece = body[start : start + TARGET_CHARS]
        if len(piece) < MIN_CHUNK_CHARS and pieces:
            break
        pieces.append((piece.strip(), offset + start))
    return pieces


def chunk_document(document: Document) -> list[Chunk]:
    """Turn one document into its chunks."""
    chunks: list[Chunk] = []
    for heading, body, offset in _sections(document.text):
        for piece, piece_offset in _windows(body, offset):
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            digest = hashlib.sha1(
                f"{document.doc_id}:{piece_offset}".encode(), usedforsecurity=False
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=digest,
                    doc_id=document.doc_id,
                    source=document.source,
                    title=document.title,
                    heading=heading,
                    text=piece,
                    url=document.url,
                    licence=document.licence,
                    offset=piece_offset,
                )
            )
    return chunks


def chunk_all(documents: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks
