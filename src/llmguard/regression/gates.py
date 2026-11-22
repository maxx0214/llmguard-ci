"""Absolute score and operational limits shared by the generic evaluators."""
from __future__ import annotations

from collections.abc import Mapping

from ..output import finite_number, nonnegative


LOWER_IS_BETTER = {"latency_ms", "latency_p95_ms", "cost", "error_rate"}
_LIMITS = {"max_latency_ms": "latency_ms", "max_cost": "cost", "max_error_rate": "error_rate"}


def degradation(before: float, after: float, *, lower_is_better: bool = False) -> float:
    return after - before if lower_is_better else before - after


def exceeds_limit(value: float, limit: float) -> bool:
    return value > limit + 1e-12


def _threshold(value, *, minimum: bool = False, unit_interval: bool = False, signed: bool = False) -> dict:
    levels = dict(value) if isinstance(value, Mapping) else {"fail": value}
    valid = finite_number if signed else nonnegative
    if (not levels or levels.keys() - {"warn", "fail"}
            or any(not valid(bound) or (unit_interval and bound > 1) for bound in levels.values())):
        raise ValueError("Invalid warn/fail bounds for this gate")
    if "warn" in levels and "fail" in levels:
        warn, fail = levels["warn"], levels["fail"]
        if (warn < fail if minimum else warn > fail):
            raise ValueError("Gate warning bound must be reached before its failure bound")
    return levels


class GatePolicy:
    def __init__(self, config: Mapping | None, quality_metrics: set[str], custom_metrics: Mapping[str, str] | None = None):
        if config is None:
            config = {}
        if not isinstance(config, Mapping) or set(config) - {"min_scores", "max_scores", "max_drop", "on_unknown", *_LIMITS}:
            raise ValueError("Invalid quality_gate settings")
        custom_metrics = dict(custom_metrics or {})
        lower_custom = {name for name, direction in custom_metrics.items() if direction == "lower_is_better"}
        self.lower = LOWER_IS_BETTER | lower_custom
        self.on_unknown = config.get("on_unknown", "fail")
        if self.on_unknown not in ("fail", "warn"):
            raise ValueError("quality_gate.on_unknown must be 'fail' or 'warn'")
        floors = config.get("min_scores", {})
        ceilings = config.get("max_scores", {})
        drops = config.get("max_drop", {})
        if not isinstance(floors, Mapping) or set(floors) - (quality_metrics | (custom_metrics.keys() - lower_custom)):
            raise ValueError("min_scores must refer to selected higher-is-better metrics")
        if not isinstance(ceilings, Mapping) or set(ceilings) - lower_custom:
            raise ValueError("max_scores must refer to selected lower-is-better custom metrics")
        if not isinstance(drops, Mapping) or set(drops) - (quality_metrics | LOWER_IS_BETTER | custom_metrics.keys()):
            raise ValueError("max_drop must refer to selected quality metrics or operational metrics")
        self.floors = {name: _threshold(rule, minimum=True, unit_interval=name in quality_metrics, signed=name in custom_metrics)
                       for name, rule in floors.items()}
        self.ceilings = {name: _threshold(rule, signed=True) for name, rule in ceilings.items()}
        self.drops = {name: _threshold(drops.get(name, 0), unit_interval=name in quality_metrics | {"error_rate"})
                      for name in sorted(quality_metrics | set(drops))}
        self.limits = {metric: _threshold(config[setting], unit_interval=metric == "error_rate")
                       for setting, metric in _LIMITS.items() if setting in config}
        self.limits.setdefault("error_rate", {"fail": 0})

    def settings(self) -> dict:
        return {"min_scores": self.floors, "max_drop": self.drops, "on_unknown": self.on_unknown,
                **({"max_scores": self.ceilings} if self.ceilings else {}),
                **{setting: self.limits[metric] for setting, metric in _LIMITS.items() if metric in self.limits}}

    def _check(self, value: float | None, rule: dict, *, minimum: bool = False) -> str:
        if value is None:
            return self.on_unknown.upper() if rule else "UNKNOWN"
        for level in ("fail", "warn"):
            if level in rule:
                limit = rule[level]
                breached = exceeds_limit(limit, value) if minimum else exceeds_limit(value, limit)
                if breached:
                    return level.upper()
        return "PASS"

    def assess(self, observed: dict, previous: dict | None = None) -> tuple[str, dict, dict]:
        checks = {}
        for limits, minimum in ((self.floors, True), ({**self.limits, **self.ceilings}, False)):
            for name, rule in limits.items():
                value = observed[name]
                checks[name] = {"value": value, "thresholds": rule, "direction": "minimum" if minimum else "maximum",
                                "status": self._check(value, rule, minimum=minimum)}
        comparison = {}
        if previous is not None:
            for name, current in observed.items():
                old = previous[name]
                lower = name in self.lower
                rule = self.drops.get(name, {})
                change = degradation(old, current, lower_is_better=lower) if old is not None and current is not None else None
                comparison[name] = {"baseline": old, "candidate": current, "delta": None if change is None else current - old,
                                    "direction": "lower" if lower else "higher", "thresholds": rule,
                                    "status": self._check(change, rule)}
        statuses = {item["status"] for item in (*checks.values(), *comparison.values())}
        status = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
        return status, checks, comparison
