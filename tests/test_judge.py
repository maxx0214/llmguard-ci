import asyncio
from unittest.mock import AsyncMock

import pytest
from conftest import case_result
from llmguard.config import JudgeConfig
from llmguard.eval.intent_judge import analyze_cases, select_judge_cases
from llmguard.llm.offline import OfflineClient
from llmguard.models import IntentJudgeResult, JudgeCaseResult
from llmguard.regression.comparator import compare
from test_classifier import fake_openai,response


def judgment(score=2):
    return IntentJudgeResult(semantic_fit=score,severity="moderate",ambiguity="medium",
        error_type="wrong_card_state",explanation="The prediction misses the card state.")


@pytest.mark.parametrize("field,value",[("semantic_fit",0),("semantic_fit",6),("semantic_fit",2.5),
    ("severity","severe"),("ambiguity","unknown"),("error_type","whatever"),("explanation","")])
def test_judge_schema(field,value):
    data = judgment().model_dump()
    data[field]=value
    with pytest.raises(ValueError):
        IntentJudgeResult.model_validate(data)


def test_judge_disabled_and_unchanged_success(make_run):
    baseline = make_run(baseline=True)
    assert select_judge_cases(baseline.cases,baseline,JudgeConfig(),42) == []
    failed = [case_result(predicted="card_linking")]
    assert select_judge_cases(failed,None,JudgeConfig(enabled=False),42) == []
    assert select_judge_cases(failed,None,JudgeConfig(max_cases=0),42) == []


def test_priority_limit_and_determinism(make_run):
    before=[case_result(f"BANKING77-TEST-{i:04d}",predicted="card_arrival" if i==3 else "card_linking")
            for i in range(1,8)]
    after=[case_result(f"BANKING77-TEST-{i:04d}",predicted="card_linking") for i in range(1,8)]
    baseline=make_run(before,True)
    config=JudgeConfig(max_cases=2)
    chosen=select_judge_cases(after,baseline,config,42)
    assert len(chosen)==2 and chosen[0].test_case_id=="BANKING77-TEST-0003"
    assert chosen==select_judge_cases(after,baseline,config,42)


def test_improvements_and_changed_failures_selected(make_run):
    before=[case_result(predicted="card_linking"),case_result("BANKING77-TEST-0002",predicted="card_linking")]
    after=[case_result(),case_result("BANKING77-TEST-0002",predicted="age_limit")]
    assert len(select_judge_cases(after,make_run(before,True),JudgeConfig(),42))==2


def test_judge_temperature_and_payload():
    client,generate=fake_openai([response(judgment().model_dump_json())])
    reply=asyncio.run(client.judge_intent("ground truth authoritative",{"query":"x"},"model"))
    assert reply.value==judgment()
    assert generate.call_args.kwargs["temperature"] == 0


def test_disabled_judge_makes_no_calls(pricing):
    client=AsyncMock()
    results,count=asyncio.run(analyze_cases([case_result()],None,client,pricing,"openai","fixture-model",
        JudgeConfig(enabled=False),42,3))
    assert results==[] and count==0
    client.judge_intent.assert_not_called()


def test_paired_optional_gate(make_run,thresholds):
    baseline,candidate=make_run(baseline=True),make_run()
    baseline.judgments=[JudgeCaseResult(test_case_id=baseline.cases[0].test_case_id,result=judgment(5))]
    candidate.judgments=[JudgeCaseResult(test_case_id=candidate.cases[0].test_case_id,result=judgment(2))]
    thresholds.judge_semantic_fit.enabled_as_gate=True
    assert compare(baseline,candidate,thresholds).status=="FAIL"

