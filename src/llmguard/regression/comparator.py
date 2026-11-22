"""Ground-truth quality gates with independent operational and optional judge gates."""
from math import isclose
from statistics import mean

from llmguard.models import EvaluationRun, MetricDelta, RegressionResult, Status, TestCaseResult
from llmguard.regression.thresholds import Thresholds


def ensure_compatible(baseline: EvaluationRun, candidate: EvaluationRun) -> None:
    if baseline.prompt.id != candidate.prompt.id:
        raise ValueError("Baseline feature/prompt ID does not match candidate")
    if baseline.dataset != candidate.dataset:
        raise ValueError("Baseline dataset/split/tier/seed/sample fingerprint does not match")
    if baseline.metadata.get("mode") != candidate.metadata.get("mode"):
        raise ValueError("Offline fixtures cannot be compared with live runs")
    if not baseline.is_baseline or baseline.metrics.error_count or baseline.metrics.invalid_predictions:
        raise ValueError("Selected run is not an error-free designated baseline")
    if baseline.metadata.get("metric_contract") != candidate.metadata.get("metric_contract"):
        raise ValueError("Metric contract changed; create a new baseline")


def case_transitions(before: list[TestCaseResult], after: list[TestCaseResult]) -> dict[str, list[str]]:
    previous = {c.test_case_id: c for c in before}
    if len(previous) != len(before) or set(previous) != {c.test_case_id for c in after} or len(before) != len(after):
        raise ValueError("Baseline and candidate test IDs must match uniquely")
    result = {key: [] for key in ("regressions", "improvements", "changed_failures", "stable_failures", "stable_successes")}
    for case in after:
        old = previous[case.test_case_id]
        if (old.input_text, old.expected_intent) != (case.input_text, case.expected_intent):
            raise ValueError("Case text or ground truth changed")
        if old.correct and not case.correct:
            kind = "regressions"
        elif not old.correct and case.correct:
            kind = "improvements"
        elif old.correct and case.correct:
            kind = "stable_successes"
        elif old.predicted_intent != case.predicted_intent:
            kind = "changed_failures"
        else:
            kind = "stable_failures"
        result[kind].append(case.test_case_id)
    return result


def severity(degradation: float, warning: float, critical: float) -> Status:
    if degradation >= critical or isclose(degradation, critical, rel_tol=0, abs_tol=1e-12):
        return "FAIL"
    if degradation >= warning or isclose(degradation, warning, rel_tol=0, abs_tol=1e-12):
        return "WARN"
    return "PASS"


def metric_delta(name: str, before: float | None, after: float | None,
                 warning: float | None = None, critical: float | None = None,
                 increase: bool = False, note: str = "") -> MetricDelta:
    if before is None or after is None:
        return MetricDelta(name=name, baseline=before, candidate=after, delta=None, severity="N/A", note=note or "Unavailable")
    delta = after - before
    pct = delta / before * 100 if before else (0 if after == 0 else None)
    level = "PASS"
    if warning is not None and critical is not None:
        degradation = pct if increase else -delta
        level = "FAIL" if degradation is None else severity(degradation, warning, critical)
    return MetricDelta(name=name, baseline=before, candidate=after, delta=delta,
                       change_pct=pct, severity=level, note=note)


def compare(baseline: EvaluationRun, candidate: EvaluationRun, thresholds: Thresholds) -> RegressionResult:
    ensure_compatible(baseline, candidate)
    transitions = case_transitions(baseline.cases, candidate.cases)
    deltas = []
    for name in ("accuracy", "macro_f1"):
        rule = getattr(thresholds, name)
        deltas.append(metric_delta(name, getattr(baseline.metrics, name), getattr(candidate.metrics, name),
                                   rule.warning_drop, rule.critical_drop))
    deltas.append(metric_delta("weighted_f1", baseline.metrics.weighted_f1, candidate.metrics.weighted_f1,
                               note="Informational; primary gates are accuracy and macro F1"))
    rule = thresholds.latency_p95
    deltas.append(metric_delta("latency_p95", baseline.metrics.latency_p95, candidate.metrics.latency_p95,
                               rule.warning_increase_pct, rule.critical_increase_pct, increase=True))
    same_pricing = baseline.metadata.get("pricing") == candidate.metadata.get("pricing")
    rule = thresholds.cost
    cost = metric_delta("estimated_cost", baseline.metrics.estimated_cost, candidate.metrics.estimated_cost,
                        rule.warning_increase_pct if same_pricing else None,
                        rule.critical_increase_pct if same_pricing else None, increase=True,
                        note="Classifier-only cost; judge spend reported separately")
    if not same_pricing:
        cost.severity, cost.note = "N/A", "Pricing configuration changed; cost gate not comparable"
    deltas.append(cost)
    judge_rule = thresholds.judge_semantic_fit
    if judge_rule.enabled_as_gate:
        for key in ("judge_model", "judge_provider", "judge_prompt_hash"):
            if baseline.metadata.get(key) != candidate.metadata.get(key):
                raise ValueError(f"Judge gate requires matching {key}")
        previous = {j.test_case_id: j.result for j in baseline.judgments if j.result}
        current = {j.test_case_id: j.result for j in candidate.judgments if j.result}
        matched = sorted(previous.keys() & current.keys())
        if not matched:
            raise ValueError("Judge gate enabled but no paired successful judgments are available")
        before = mean(previous[k].semantic_fit for k in matched)
        after = mean(current[k].semantic_fit for k in matched)
        deltas.append(metric_delta("judge_semantic_fit", before, after, judge_rule.warning_drop,
                                   judge_rule.critical_drop, note=f"Paired judgments only: {len(matched)} cases"))
    else:
        deltas.append(metric_delta("judge_semantic_fit", baseline.metrics.judge_semantic_fit,
                                   candidate.metrics.judge_semantic_fit, note="Descriptive only; selected cohorts may differ"))
    reasons = [f"{d.name}: {d.severity}" for d in deltas if d.severity in ("WARN", "FAIL")]
    levels = [d.severity for d in deltas]
    errors = candidate.metrics.invalid_predictions + candidate.metrics.error_count
    if errors / max(candidate.metrics.total_cases, 1) > thresholds.max_error_rate or not candidate.metrics.valid_predictions:
        levels.append("FAIL")
        reasons.append(f"Operational error gate: {errors}/{candidate.metrics.total_cases}")
    status: Status = "FAIL" if "FAIL" in levels else "WARN" if "WARN" in levels else "PASS"
    return RegressionResult(baseline_run_id=baseline.run_id, candidate_run_id=candidate.run_id,
                            status=status, metrics=deltas, transitions=transitions, reasons=reasons)

