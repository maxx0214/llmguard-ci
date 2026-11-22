import pytest
from conftest import case_result
from llmguard.eval.metrics import compute_metrics, percentile


def test_accuracy_macro_weighted_and_per_class():
    cases = [case_result(), case_result("BANKING77-TEST-0002"),
             case_result("BANKING77-TEST-0003", expected="card_linking", predicted="card_arrival")]
    m = compute_metrics(cases)
    assert m.accuracy == pytest.approx(2/3)
    assert m.per_class["card_arrival"].precision == pytest.approx(2/3)
    assert m.per_class["card_arrival"].recall == 1
    assert m.per_class["card_arrival"].f1 == pytest.approx(.8)
    assert m.per_class["card_arrival"].support == 2
    assert m.macro_f1 == pytest.approx(.8/77)
    assert m.macro_precision == pytest.approx((2/3)/77)
    assert m.macro_recall == pytest.approx(1/77)
    assert m.weighted_f1 == pytest.approx(.8 * 2/3)
    assert sum(map(sum, m.confusion_matrix)) == 3
    assert m.top_confusions[0].expected_intent == "card_linking"


def test_invalid_and_execution_errors_cannot_inflate_accuracy():
    cases = [case_result(), case_result("BANKING77-TEST-0002", predicted=None, error="bad",kind="invalid_output"),
             case_result("BANKING77-TEST-0003",predicted=None,error="timeout",kind="execution")]
    m = compute_metrics(cases)
    assert m.accuracy == pytest.approx(1/3)
    assert (m.valid_predictions,m.invalid_predictions,m.error_count) == (1,1,1)
    assert sum(row[-1] for row in m.confusion_matrix) == 2
    assert m.error_rate == pytest.approx(2/3)


def test_empty_and_percentiles():
    assert compute_metrics([]).accuracy == 0
    assert percentile([], .95) == 0
    assert percentile([10], .95) == 10
    assert percentile([10,20,30,40], .95) == pytest.approx(38.5)


def test_unknown_cost_propagates():
    case = case_result()
    case.estimated_cost = None
    m = compute_metrics([case])
    assert m.estimated_cost is None and m.total_estimated_cost is None

