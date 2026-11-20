"""Atomic snapshots and an append-only, per-run event journal."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from .schemas import Claim, DocumentVersion, Passage


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        journal = root / "trajectory.jsonl"
        self.sequence = len(journal.read_text().splitlines()) if journal.exists() else 0

    def event(self, kind: str, **fields) -> None:
        with self._lock:
            self.sequence += 1
            record = {
                "sequence": self.sequence,
                "time": datetime.now(UTC).isoformat(),
                "kind": kind,
                **fields,
            }
            with (self.root / "trajectory.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def save(self, name: str, value) -> None:
        with self._lock:
            write_json(self.root / name, value)


class EvidenceStore:
    def __init__(self, store: RunStore):
        self.store = store
        path = store.root / "evidence.json"
        data = read_json(path) if path.exists() else {}
        self.documents = {
            k: DocumentVersion.model_validate(v) for k, v in data.get("documents", {}).items()
        }
        for key, expected in data.get("document_hashes", {}).items():
            if key not in self.documents or self.documents[key].content_hash != expected:
                raise ValueError("saved evidence content hash mismatch")
        self.passages = {k: Passage.model_validate(v) for k, v in data.get("passages", {}).items()}
        for passage in self.passages.values():
            doc = self.documents.get(passage.version_id)
            if doc is None or doc.content[passage.start : passage.end] != passage.text:
                raise ValueError("saved passage does not match original offsets")
        claims = store.root / "claims.json"
        self.claims = (
            [Claim.model_validate(c) for c in read_json(claims)] if claims.exists() else []
        )

    def add_document(self, document: DocumentVersion) -> list[Passage]:
        existing = self.documents.get(document.version_id)
        if existing and existing.model_dump() != document.model_dump():
            raise ValueError("immutable version ID reused with different content or metadata")
        self.documents[document.version_id] = document
        # Offsets refer to the exact saved content, including paragraph separators.
        offset = 0
        added = []
        for paragraph in document.content.split("\n\n"):
            if paragraph.strip():
                for within in range(0, len(paragraph), 1500):
                    part = paragraph[within : within + 1500]
                    start = offset + within
                    added.append(self._passage(document, start, start + len(part)))
            offset += len(paragraph) + 2
        self.persist()
        return added

    def _passage(self, document, start, end):
        identity = f"{document.version_id}:{document.content_hash}:{start}:{end - start}"
        pid = "p-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        passage = Passage(
            passage_id=pid,
            version_id=document.version_id,
            start=start,
            end=end,
            text=document.content[start:end],
        )
        self.passages[pid] = passage
        return passage

    def add_source_spans(self, version_id, spans):
        document = self.documents[version_id]
        spans = list(dict.fromkeys(spans))
        if any(not 0 <= start < end <= len(document.content) for start, end in spans):
            raise ValueError("source span exceeds saved document")
        passages = {span: self._passage(document, *span) for span in spans}
        self.persist()
        return passages

    def add_claim(
        self, text: str, passage_ids: list[str], event_time: str | None, stage: str | None
    ) -> Claim:
        if not passage_ids or any(pid not in self.passages for pid in passage_ids):
            raise ValueError("claims must reference passages already read in this run")
        identity = json.dumps([text, sorted(set(passage_ids)), event_time, stage])
        cid = "c-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        previous = next((c for c in self.claims if c.claim_id == cid), None)
        if previous:
            return previous
        claim = Claim(
            claim_id=cid,
            text=text,
            passage_ids=sorted(set(passage_ids)),
            event_time=event_time,
            stage=stage,
        )
        self.claims.append(claim)
        self.persist()
        return claim

    def persist(self) -> None:
        self.store.save(
            "evidence.json",
            {
                "documents": {k: v.model_dump(mode="json") for k, v in self.documents.items()},
                "document_hashes": {k: v.content_hash for k, v in self.documents.items()},
                "passages": {k: v.model_dump() for k, v in self.passages.items()},
            },
        )
        self.store.save("claims.json", [c.model_dump() for c in self.claims])
