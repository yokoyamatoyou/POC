"""要約成功率・フォールバック率を収集するユーティリティ"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

LOG_PATH = Path("logs/summary_metrics.jsonl")


@dataclass
class SummaryMetricsEvent:
    operation: str
    outcome: str  # success | failure | fallback
    timestamp: float
    detail: Dict[str, Any]


class SummaryMetricsTracker:
    """要約処理の成功/失敗/フォールバックを集計する"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, Dict[str, int]] = {}
        self._recent: list[SummaryMetricsEvent] = []

    def record(self, operation: str, outcome: str, detail: Optional[Dict[str, Any]] = None) -> None:
        detail = detail or {}
        event = SummaryMetricsEvent(
            operation=operation,
            outcome=outcome,
            timestamp=time.time(),
            detail=detail,
        )

        with self._lock:
            op_counts = self._counts.setdefault(operation, {})
            op_counts[outcome] = op_counts.get(outcome, 0) + 1

            self._recent.append(event)
            if len(self._recent) > 500:
                self._recent = self._recent[-500:]

        self._append_to_log(event)

    def get_counts(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return json.loads(json.dumps(self._counts))  # deep copy via json

    def get_recent(self) -> list[SummaryMetricsEvent]:
        with self._lock:
            return list(self._recent)

    def _append_to_log(self, event: SummaryMetricsEvent) -> None:
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        except Exception:
            # ログ出力失敗は無視（metrics 収集に影響させない）
            pass


_tracker = SummaryMetricsTracker()


def record_summary_event(operation: str, outcome: str, detail: Optional[Dict[str, Any]] = None) -> None:
    _tracker.record(operation, outcome, detail)


def get_summary_metrics() -> Dict[str, Dict[str, int]]:
    return _tracker.get_counts()


def get_recent_summary_events() -> list[SummaryMetricsEvent]:
    return _tracker.get_recent()








