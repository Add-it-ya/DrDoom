"""Record real provider responses once, replay them forever after.

An evaluation suite that calls a live model is one nobody runs: it costs money on every
push, it fails when a key expires, and it gives a different answer each time, so a
regression and ordinary variance look identical. Recording the responses once and
replaying them makes the suite free, offline, and deterministic -- which is what lets it
sit in continuous integration and actually gate a merge.

The trade is real and worth stating: a replayed suite measures whether the *pipeline*
changed, not whether the model did. Prompts, retrieval, parsing and validation are all
under test; the model's behaviour is frozen at the moment of recording. Re-record when the
prompts change, and read the diff.

Snapshots are keyed by a hash of everything that determines a response -- model, system
prompt, messages, schema -- so changing a prompt misses the cache rather than silently
replaying an answer to a different question.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from drdoom.llm.base import Completion, LLMProvider, LLMUnavailableError, Message

logger = logging.getLogger(__name__)


def request_key(
    model: str, messages: list[Message], system: str | None, json_schema: dict | None
) -> str:
    """A stable fingerprint of everything that decides the response."""
    payload = {
        "model": model,
        "system": system or "",
        "messages": [{"role": message.role, "content": message.content} for message in messages],
        "schema": json_schema or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


class SnapshotStore:
    """One json file per recorded response, named by its request key.

    A manifest records which model produced the set. Without it, replay has to guess the
    model that goes into the key, guesses wrong, misses every lookup, and the pipeline
    degrades quietly to its no-model path -- which scores badly and looks like a quality
    regression rather than a configuration mistake.
    """

    MANIFEST = "manifest.json"

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def recorded_model(self) -> str | None:
        manifest = self.directory / self.MANIFEST
        if not manifest.is_file():
            return None
        return json.loads(manifest.read_text(encoding="utf-8")).get("model")

    def set_recorded_model(self, model: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / self.MANIFEST).write_text(
            json.dumps({"model": model}, indent=2), encoding="utf-8"
        )

    def read(self, key: str) -> Completion | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        return Completion(
            text=record["text"],
            model=record["model"],
            provider=record["provider"],
            input_tokens=record.get("input_tokens", 0),
            output_tokens=record.get("output_tokens", 0),
        )

    def write(self, key: str, completion: Completion, note: str = "") -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path_for(key).write_text(
            json.dumps(
                {
                    "note": note,
                    "provider": completion.provider,
                    "model": completion.model,
                    "input_tokens": completion.input_tokens,
                    "output_tokens": completion.output_tokens,
                    "text": completion.text,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        if not self.directory.is_dir():
            return 0
        return len([p for p in self.directory.glob("*.json") if p.name != self.MANIFEST])


class RecordingProvider:
    """Calls a real provider and saves what it said."""

    def __init__(self, inner: LLMProvider, store: SnapshotStore, note: str = "") -> None:
        self.inner = inner
        self.store = store
        self.note = note
        self.name = f"recording[{inner.name}]"
        self.model = inner.model

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        completion = self.inner.complete(messages, **kwargs)
        key = request_key(self.model, messages, kwargs.get("system"), kwargs.get("json_schema"))
        self.store.set_recorded_model(self.model)
        self.store.write(key, completion, note=self.note)
        logger.info("recorded %s", key)
        return completion


class ReplayProvider:
    """Serves recorded responses, and refuses to invent one it does not have."""

    def __init__(self, store: SnapshotStore, model: str | None = None) -> None:
        self.store = store
        self.name = "replay"
        # The model is part of the request key, so it has to be the one that was
        # recorded rather than a placeholder.
        self.model = model or store.recorded_model() or "replay"
        self.misses: list[str] = []

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        key = request_key(self.model, messages, kwargs.get("system"), kwargs.get("json_schema"))
        completion = self.store.read(key)
        if completion is None:
            self.misses.append(key)
            raise LLMUnavailableError(
                f"no recorded response for {key}; re-record with a configured provider"
            )
        return completion
