"""ACL監査ログを読み込むユーティリティ"""

from __future__ import annotations

import csv
import io
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_AUDIT_PATH = Path(os.getenv("ACL_AUDIT_LOG_PATH", "logs/access_audit.log"))


@dataclass
class AuditEvent:
    timestamp: str
    doc_id: Optional[str]
    tenant_id: Optional[str]
    access_scope: Optional[str]
    min_role_level: Optional[int]
    decision: str
    reason: str
    user: dict

    @classmethod
    def from_json(cls, raw: str) -> "AuditEvent":
        data = json.loads(raw)
        return cls(
            timestamp=data.get("timestamp", ""),
            doc_id=data.get("doc_id"),
            tenant_id=data.get("tenant_id"),
            access_scope=data.get("access_scope"),
            min_role_level=data.get("min_role_level"),
            decision=data.get("decision", "UNKNOWN"),
            reason=data.get("reason", ""),
            user=data.get("user") or {},
        )


def _iter_events(path: Path) -> Iterable[AuditEvent]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield AuditEvent.from_json(line)
            except json.JSONDecodeError:
                continue


def _filter_events(
    events: Sequence[AuditEvent],
    tenant_id: Optional[str],
    decision: Optional[str],
) -> List[AuditEvent]:
    filtered: List[AuditEvent] = []
    for event in events:
        if tenant_id and event.tenant_id != tenant_id:
            continue
        if decision and event.decision != decision:
            continue
        filtered.append(event)
    return filtered


def load_events(
    limit: int = 100,
    *,
    tenant_id: Optional[str] = None,
    decision: Optional[str] = None,
    path: Path = DEFAULT_AUDIT_PATH,
) -> List[AuditEvent]:
    events = list(reversed(list(_iter_events(path))))
    filtered = _filter_events(events, tenant_id, decision)
    return filtered[:limit]


def load_events_paginated(
    page: int,
    page_size: int,
    *,
    tenant_id: Optional[str] = None,
    decision: Optional[str] = None,
    path: Path = DEFAULT_AUDIT_PATH,
) -> Tuple[List[AuditEvent], int]:
    page = max(page, 1)
    page_size = max(page_size, 1)

    events = list(reversed(list(_iter_events(path))))
    filtered = _filter_events(events, tenant_id, decision)
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_events = filtered[start:end]
    if start >= total and total > 0:
        last_page = (total - 1) // page_size + 1
        start = (last_page - 1) * page_size
        end = start + page_size
        page_events = filtered[start:end]
    return page_events, total


def events_to_csv(events: Sequence[AuditEvent]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "timestamp",
        "doc_id",
        "tenant_id",
        "access_scope",
        "min_role_level",
        "decision",
        "reason",
        "user",
    ])
    for event in events:
        writer.writerow([
            event.timestamp,
            event.doc_id or "",
            event.tenant_id or "",
            event.access_scope or "",
            event.min_role_level if event.min_role_level is not None else "",
            event.decision,
            event.reason,
            json.dumps(event.user, ensure_ascii=False),
        ])
    return buffer.getvalue()


def summarize_decisions(events: Iterable[AuditEvent]) -> Counter:
    counter: Counter[str] = Counter()
    for event in events:
        counter[event.decision] += 1
    return counter








