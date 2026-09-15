"""Authoritative ground-truth metrics; judge output never alters correctness."""
from collections import Counter
from statistics import mean

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from llmguard.data.intents import INTENTS
from llmguard.models import ClassMetrics, ConfusionPair, JudgeCaseResult, Metrics, TestCaseResult


def percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation, including singleton and empty sets."""
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def sum_cost(values: list[float | None]) -> float | None:
    return None if any(v is None for v in values) else sum(v for v in values if v is not None)


def compute_metrics(cases: list[TestCaseResult],
                    judgments: list[JudgeCaseResult] | None = None) -> Metrics:
    judgments = judgments or []
    truth = [c.expected_intent for c in cases]
    predicted = [c.predicted_intent or "__ERROR__" for c in cases]
    labels = list(INTENTS)
    if cases:
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=labels, zero_division=0)
        accuracy = float(accuracy_score(truth, predicted))
        per_class = {label: ClassMetrics(support=int(s), precision=float(p), recall=float(r), f1=float(f))
                     for label, p, r, f, s in zip(labels, precision, recall, f1, support)}
        macro_p, macro_r, macro_f = float(precision.mean()), float(recall.mean()), float(f1.mean())
        weighted = float((f1 * support).sum() / len(cases))
    else:
        accuracy = macro_p = macro_r = macro_f = weighted = 0.0
        per_class = {label: ClassMetrics(support=0, precision=0, recall=0, f1=0) for label in labels}
    # Include an error column so invalid/failed requests do not disappear.
    matrix_labels = labels + ["__ERROR__"]
    matrix = confusion_matrix(truth, predicted, labels=matrix_labels).tolist() if cases else [[0]*78 for _ in range(78)]
    pairs = Counter((a, b) for a, b in zip(truth, predicted) if a != b)
    latency = [c.latency_ms for c in cases]
    analyzed = [j.result for j in judgments if j.result is not None]
    classifier_cost = sum_cost([c.estimated_cost for c in cases])
    judge_cost = sum_cost([j.estimated_cost for j in judgments])
    return Metrics(total_cases=len(cases),
        valid_predictions=sum(c.predicted_intent is not None for c in cases),
        invalid_predictions=sum(c.error_kind == "invalid_output" for c in cases),
        error_count=sum(c.error_kind == "execution" for c in cases), error_rate=1-accuracy if cases else 0,
        accuracy=accuracy, macro_precision=macro_p, macro_recall=macro_r, macro_f1=macro_f,
        weighted_f1=weighted, per_class=per_class, confusion_labels=matrix_labels, confusion_matrix=matrix,
        top_confusions=[ConfusionPair(expected_intent=a, predicted_intent=b, count=n)
                        for (a, b), n in sorted(pairs.items(), key=lambda x: (-x[1], x[0]))[:20]],
        latency_mean=mean(latency) if latency else 0, latency_p50=percentile(latency, .5),
        latency_p95=percentile(latency, .95), input_tokens=sum(c.input_tokens for c in cases),
        output_tokens=sum(c.output_tokens for c in cases), estimated_cost=classifier_cost,
        judge_count=len(analyzed), judge_error_count=sum(j.error is not None for j in judgments),
        judge_semantic_fit=mean(j.semantic_fit for j in analyzed) if analyzed else None,
        judge_severity_counts=dict(Counter(j.severity for j in analyzed)),
        judge_input_tokens=sum(u.input_tokens for j in judgments for u in j.usage),
        judge_output_tokens=sum(u.output_tokens for j in judgments for u in j.usage),
        judge_cost=judge_cost, total_estimated_cost=sum_cost([classifier_cost, judge_cost]))
