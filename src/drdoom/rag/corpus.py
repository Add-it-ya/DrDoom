"""Fetch the operational documentation the diagnosis agent reasons over.

The corpus is real vendor documentation, not runbooks written for this project. Four
documents authored here, one per anomaly type, would make retrieval a lookup: whatever
the classifier already decided would select the document, and semantic search would run
over a handful of chunks it could not fail to rank correctly. A few hundred documents
about overlapping subjects is what makes retrieval a real problem with a measurable
answer.

Documents are downloaded at build time and never committed, so the repository holds no
redistributed third-party content. Each document keeps its source, path, upstream URL and
licence, so any retrieved passage can be attributed.
"""

from __future__ import annotations

import json
import logging
import re
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from drdoom.config import get_settings

logger = logging.getLogger(__name__)

FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
TITLE_FIELD = re.compile(r'^title:\s*"?(.+?)"?\s*$', re.MULTILINE)
HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)
SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
BLANK_RUN = re.compile(r"\n{3,}")

MIN_DOCUMENT_CHARS = 400


@dataclass(frozen=True)
class SourceSpec:
    """Where a slice of the corpus comes from and under what licence."""

    name: str
    archive_url: str
    include: tuple[str, ...]
    licence: str
    url_prefix: str
    strip_prefix: str


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="prometheus",
        archive_url="https://codeload.github.com/prometheus/docs/tar.gz/refs/heads/main",
        include=("docs/", "blog/posts/"),
        licence="Apache-2.0",
        url_prefix="https://github.com/prometheus/docs/blob/main/",
        strip_prefix="docs-main/",
    ),
    SourceSpec(
        name="kubernetes",
        archive_url="https://codeload.github.com/kubernetes/website/tar.gz/refs/heads/main",
        include=(
            "content/en/docs/tasks/",
            "content/en/docs/concepts/cluster-administration/",
            "content/en/docs/concepts/workloads/",
            "content/en/docs/reference/kubectl/",
        ),
        licence="CC-BY-4.0",
        url_prefix="https://github.com/kubernetes/website/blob/main/",
        strip_prefix="website-main/",
    ),
)


@dataclass(frozen=True)
class Document:
    """One documentation page, with enough provenance to cite it."""

    doc_id: str
    source: str
    path: str
    title: str
    text: str
    url: str
    licence: str


def corpus_dir() -> Path:
    return get_settings().raw_data_dir / "corpus"


def clean_markdown(raw: str) -> tuple[str, str]:
    """Strip frontmatter and template noise, returning ``(title, body)``."""
    title = ""
    match = FRONTMATTER.match(raw)
    if match:
        found = TITLE_FIELD.search(match.group(1))
        if found:
            title = found.group(1).strip()
        raw = raw[match.end() :]

    body = SHORTCODE.sub(" ", raw)
    body = HTML_COMMENT.sub(" ", body)
    body = BLANK_RUN.sub("\n\n", body).strip()

    if not title:
        heading = HEADING.search(body)
        title = heading.group(1).strip() if heading else ""
    return title, body


def _wanted(member_name: str, spec: SourceSpec) -> str | None:
    """Return the repository-relative path when a member belongs in the corpus."""
    if not member_name.endswith(".md") or ".." in member_name:
        return None
    relative = (
        member_name[len(spec.strip_prefix) :]
        if member_name.startswith(spec.strip_prefix)
        else member_name
    )
    if not any(relative.startswith(prefix) for prefix in spec.include):
        return None
    if relative.endswith("_index.md"):
        return None
    return relative


def fetch_source(spec: SourceSpec) -> list[Document]:
    """Stream one archive and keep the documentation pages it contains."""
    logger.info("fetching %s corpus", spec.name)
    documents: list[Document] = []
    with (
        urllib.request.urlopen(spec.archive_url) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        for member in archive:
            if not member.isfile():
                continue
            relative = _wanted(member.name, spec)
            if relative is None:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            raw = handle.read().decode("utf-8", errors="replace")
            title, body = clean_markdown(raw)
            if len(body) < MIN_DOCUMENT_CHARS or not title:
                continue
            documents.append(
                Document(
                    doc_id=f"{spec.name}:{relative}",
                    source=spec.name,
                    path=relative,
                    title=title,
                    text=body,
                    url=spec.url_prefix + relative,
                    licence=spec.licence,
                )
            )
    logger.info("%s: kept %d documents", spec.name, len(documents))
    return documents


def download(sources: tuple[SourceSpec, ...] = SOURCES, force: bool = False) -> Path:
    """Fetch every source and cache the result as one json file."""
    path = corpus_dir() / "documents.json"
    if path.is_file() and not force:
        logger.info("corpus already present at %s", path)
        return path

    documents: list[Document] = []
    for spec in sources:
        documents.extend(fetch_source(spec))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(document) for document in documents], indent=1), encoding="utf-8"
    )
    logger.info("wrote %d documents to %s", len(documents), path)
    return path


def load() -> list[Document]:
    path = corpus_dir() / "documents.json"
    if not path.is_file():
        raise FileNotFoundError(f"corpus not downloaded; expected {path}")
    return [Document(**record) for record in json.loads(path.read_text(encoding="utf-8"))]


def is_downloaded() -> bool:
    return (corpus_dir() / "documents.json").is_file()
