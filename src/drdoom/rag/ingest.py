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
import html
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


# The elements documentation pages actually use. Anything else in angle brackets, such as
# <namespace> in "kubectl get pods -n <namespace>", is a placeholder the reader needs.
BLOCK_ELEMENTS = frozenset(
    {
        "blockquote", "br", "caption", "dd", "details", "div", "dl", "dt", "figcaption",
        "figure", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "li", "ol", "p", "pre",
        "section", "summary", "table", "tbody", "tfoot", "thead", "tr", "ul",
    }
)  # fmt: skip
HTML_ELEMENTS = BLOCK_ELEMENTS | frozenset(
    {
        "a", "abbr", "audio", "b", "center", "cite", "code", "col", "colgroup", "del",
        "em", "embed", "font", "i", "iframe", "img", "input", "ins", "kbd", "mark",
        "object", "q", "s", "samp", "small", "source", "span", "strong", "sub", "sup",
        "td", "th", "u", "var", "video",
    }
)  # fmt: skip
# Content no reader of the rendered page sees, which is where an instruction aimed at the
# model would hide: scripts, styles and comments go with everything inside them.
HIDDEN = re.compile(
    r"<(script|style|template|noscript)\b[^>]*>.*?</\1\s*>|<!--.*?-->", re.IGNORECASE | re.DOTALL
)
TAG = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*/?>")
ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
# Escaped markup is how a page shows markup to its reader, so it stays escaped.
ESCAPED_MARKUP = frozenset({"lt", "gt", "amp"})
FENCE = re.compile(r"^\s*(```|~~~)")


def strip_html(text: str) -> str:
    """Remove the HTML from a markdown page, leaving what its reader would read.

    Retrieved passages go straight into model prompts. A fifth of the corpus's chunks
    carried HTML, mostly table markup, and two blog posts carried a script tag; markup in
    a prompt costs tokens, and markup a model is shown is markup it may reproduce.

    Code fences are left exactly as written, since what looks like a tag there is a
    placeholder or an example. A page with no markup comes back unchanged.
    """
    blocks: list[tuple[bool, list[str]]] = [(False, [])]
    for line in text.split("\n"):
        in_code = blocks[-1][0]
        if FENCE.match(line) and not in_code:
            blocks.append((True, [line]))
        elif FENCE.match(line):
            blocks[-1][1].append(line)
            blocks.append((False, []))
        else:
            blocks[-1][1].append(line)
    return "\n".join(
        "\n".join(lines) if code else _strip_prose("\n".join(lines))
        for code, lines in blocks
        if lines
    )


def _strip_prose(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1).lower()
        if name not in HTML_ELEMENTS:
            return match.group(0)
        if name in ("td", "th"):
            return "" if match.group(0).startswith("</") else " | "
        return "\n" if name in BLOCK_ELEMENTS else ""

    stripped = TAG.sub(replace, HIDDEN.sub("", text))
    if stripped == text:
        return text
    stripped = ENTITY.sub(
        lambda m: m.group(0) if m.group(1).lower() in ESCAPED_MARKUP else html.unescape(m.group(0)),
        stripped,
    )
    # Tidy what the removed markup leaves behind: trailing spaces and runs of blank lines.
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    return re.sub(r"\n{3,}", "\n\n", stripped)


def chunk_document(document: Document) -> list[Chunk]:
    """Turn one document into its chunks, from its text with the HTML taken out."""
    chunks: list[Chunk] = []
    for heading, body, offset in _sections(strip_html(document.text)):
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
