import pytest
from conftest import case_result
from llmguard.regression.comparator import case_transitions, compare


@pytest.mark.parametrize("drop,status", [(0,"PASS"),(.019,"PASS"),(.02,"WARN"),(.04,"FAIL"),(.1,"FAIL")])
@pytest.mark.parametrize("metric", ["accuracy","macro_f1"])
def test_quality_boundaries(make_run, thresholds, drop, status, metric):
    baseline, candidate = make_run(baseline=True), make_run()
    # Isolate threshold logic from metric calculation, tested independently.
    setattr(baseline.metrics, metric, .9)
    setattr(candidate.metrics, metric, .9-drop)
    assert compare(baseline,candidate,thresholds).status == status


@pytest.mark.parametrize("metric,delta,status", [("latency_p95",25,"WARN"),("latency_p95",50,"FAIL"),
    ("estimated_cost",.00025,"WARN"),("estimated_cost",.0005,"FAIL"),("latency_p95",-10,"PASS")])
def test_operational_boundaries(make_run, thresholds, metric, delta, status):
    baseline, candidate = make_run(baseline=True), make_run()
    setattr(candidate.metrics, metric, getattr(candidate.metrics,metric)+delta)
    assert compare(baseline,candidate,thresholds).status == status


def test_transitions():
    before = [case_result(),case_result("BANKING77-TEST-0002",predicted="card_linking"),
              case_result("BANKING77-TEST-0003",predicted="card_linking"),
              case_result("BANKING77-TEST-0004",predicted="card_linking"),case_result("BANKING77-TEST-0005")]
    after = [case_result(predicted="card_linking"),case_result("BANKING77-TEST-0002"),
             case_result("BANKING77-TEST-0003",predicted="age_limit"),
             case_result("BANKING77-TEST-0004",predicted="card_linking"),case_result("BANKING77-TEST-0005")]
    result = case_transitions(before,after)
    assert result == dict(regressions=["BANKING77-TEST-0001"],improvements=["BANKING77-TEST-0002"],
        changed_failures=["BANKING77-TEST-0003"],stable_failures=["BANKING77-TEST-0004"],
        stable_successes=["BANKING77-TEST-0005"])


def test_no_cost_no_false_gate(make_run, thresholds):
    baseline,candidate = make_run(baseline=True),make_run()
    candidate.metrics.estimated_cost = None
    result = compare(baseline,candidate,thresholds)
    assert result.status == "PASS"
    assert next(d for d in result.metrics if d.name=="estimated_cost").severity == "N/A"


def test_zero_baseline_cost(make_run, thresholds):
    baseline,candidate = make_run(baseline=True),make_run()
    baseline.metrics.estimated_cost = 0
    result = compare(baseline,candidate,thresholds)
    assert result.status == "FAIL"
    assert "Infinity" not in result.model_dump_json()


@pytest.mark.parametrize("field,value",[("tier","full"),("seed",1),("fingerprint","changed"),("split","train")])
def test_dataset_mismatch(make_run, thresholds, field, value):
    baseline,candidate = make_run(baseline=True),make_run()
    setattr(candidate.dataset,field,value)
    with pytest.raises(ValueError,match="dataset"):
        compare(baseline,candidate,thresholds)


def test_provider_change_allowed(make_run, thresholds):
    baseline,candidate = make_run(baseline=True),make_run()
    candidate.prompt = candidate.prompt.model_copy(update={"provider":"openai","model":"different"})
    assert compare(baseline,candidate,thresholds).status == "PASS"


def test_judge_is_nonblocking_by_default(make_run, thresholds):
    baseline,candidate = make_run(baseline=True),make_run()
    baseline.metrics.judge_semantic_fit = 5
    candidate.metrics.judge_semantic_fit = 1
    assert compare(baseline,candidate,thresholds).status == "PASS"


def test_enabled_judge_gate_requires_pairs(make_run, thresholds):
    baseline,candidate = make_run(baseline=True),make_run()
    thresholds.judge_semantic_fit.enabled_as_gate = True
    with pytest.raises(ValueError,match="paired"):
        compare(baseline,candidate,thresholds)
