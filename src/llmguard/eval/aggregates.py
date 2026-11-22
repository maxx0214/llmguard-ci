"""Shared aggregation for measured latency and optional cost."""

from collections import Counter
from statistics import mean


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


def classification_macro_f1(expected, predicted) -> float:
    """Macro F1 over the labels present in the ground truth of this sample."""
    actual_counts = Counter(expected)
    predicted_counts = Counter(predicted)
    true_positives = Counter(label for label, guess in zip(expected, predicted, strict=True) if label == guess)
    return mean(2 * true_positives[label] / (count + predicted_counts[label])
                for label, count in actual_counts.items())
