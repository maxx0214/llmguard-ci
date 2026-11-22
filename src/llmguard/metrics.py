"""User-defined per-case metrics and metadata summaries."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Metric:
    """Score a case. Return a finite number, bool, or None for unknown."""

    score: Callable[[dict[str, Any]], float | bool | None]
    direction: str = "higher_is_better"
    version: str = "1"

    def __post_init__(self):
        if not callable(self.score) or self.direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError("Metric requires a callable and a supported direction")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Metric version must be a nonempty string")


def summarize_metadata(cases: list[dict]) -> dict:
    metadata = [case.get("metadata", {}) for case in cases]
    result = {}
    for name in sorted({name for values in metadata for name in values}):
        counts = Counter(json.dumps(values[name], sort_keys=True, ensure_ascii=False)
                         for values in metadata if values.get(name) is not None)
        known = sum(counts.values())
        result[name] = {"known": known, "unknown": len(cases) - known,
                        "values": [{"value": json.loads(value), "count": count} for value, count in sorted(counts.items())]}
    return result
