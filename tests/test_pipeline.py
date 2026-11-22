import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
import yaml
from conftest import ROOT,case_result

from llmguard.cli import main
from llmguard.config import JudgeConfig
from llmguard.eval.runner import evaluate
from llmguard.llm.client import LLMError
from llmguard.llm.offline import OfflineClient,fixture_dataset
from llmguard.models import EvaluationRun,TokenUsage
from llmguard.report.html_report import write_report
from llmguard.storage.database import Database,export_baseline,read_baseline


def test_bounded_concurrency_and_usage(prompt,pricing):
    cases,data=fixture_dataset()
    class Observed(OfflineClient):
        active=0
        peak=0
        async def classify(self,query,prompt):
            self.active+=1
            self.peak=max(self.peak,self.active)
            await asyncio.sleep(.001)
            value=await super().classify(query,prompt)
            self.active-=1
            return value
    client=Observed(cases)
    run=asyncio.run(evaluate(prompt,cases,data,client,pricing,prompt.model,JudgeConfig(enabled=False),2,offline=True))
    assert client.peak==2
    assert [c.test_case_id for c in run.cases]==[c.id for c in cases]
    assert run.metrics.input_tokens==35000 and run.metrics.output_tokens==1200
    assert run.metrics.estimated_cost==pytest.approx(.0374)
    assert run.metrics.judge_count==0


def test_judge_failure_does_not_change_ground_truth(prompt,pricing):
    cases,data=fixture_dataset()
    class BrokenJudge(OfflineClient):
        async def judge_intent(self,*args):
            raise LLMError("judge timeout",[TokenUsage(provider="openai",model=prompt.model,
                input_tokens=0,output_tokens=0,complete=False)])
    run=asyncio.run(evaluate(prompt,cases,data,BrokenJudge(cases),pricing,prompt.model,offline=True))
    assert run.metrics.judge_error_count>0
    assert run.metrics.error_count==0 and run.status=="PASS"
    assert run.metrics.judge_cost is None


def test_sqlite_roundtrip_and_atomicity(tmp_path,make_run):
    db=Database(tmp_path/"history.sqlite")
    run=make_run(baseline=True)
    try:
        db.save(run)
        assert db.baseline(run.prompt.id,run.dataset,"offline")==run
        assert db.baseline(run.prompt.id,run.dataset,"offline",run.run_id)==run
        assert db.connection.execute("PRAGMA user_version").fetchone()[0]==2
        with pytest.raises(sqlite3.IntegrityError):
            db.save(run)
        assert db.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]==1
    finally:
        db.close()
    export_baseline(run,tmp_path/"baseline.json")
    assert read_baseline(tmp_path/"baseline.json")==run


def test_legacy_database_is_not_modified(tmp_path):
    path=tmp_path/"legacy.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE runs (run_id TEXT, category_accuracy REAL)")
        db.execute("INSERT INTO runs VALUES ('old',0.9)")
    before=path.read_bytes()
    with pytest.raises(ValueError,match="legacy"):
        Database(path)
    assert path.read_bytes()==before
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM runs").fetchall()==[("old",.9)]


def test_future_database_is_rejected(tmp_path):
    path=tmp_path/"future.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError,match="99"):
        Database(path)


def test_legacy_json_rejected(tmp_path):
    path=tmp_path/"old.json"
    path.write_text('{"category_accuracy":0.9}')
    with pytest.raises(ValueError,match="Legacy"):
        read_baseline(path)


def test_report_escapes_query_and_judge(tmp_path,make_run):
    from llmguard.models import IntentJudgeResult,JudgeCaseResult
    run=make_run()
    run.cases[0].input_text='<script>alert("query")</script>'
    run.judgments=[JudgeCaseResult(test_case_id=run.cases[0].test_case_id,result=IntentJudgeResult(
        semantic_fit=2,severity="moderate",ambiguity="low",error_type="other",explanation="<img src=x onerror=alert(1)>"))]
    path=tmp_path/"report.html"
    write_report(run,path)
    text=path.read_text(encoding="utf-8")
    assert "<script>alert" not in text and "&lt;script&gt;" in text
    assert "<img src=x" not in text and "&lt;img" in text
    assert "Top Confused Intent Pairs" in text


def test_cli_end_to_end_exit_codes(tmp_path,monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY",str(tmp_path/"summary.md"))
    common=["--offline","--db",str(tmp_path/"demo.sqlite"),"--output-dir",str(tmp_path)]
    baseline=tmp_path/"baseline.json"
    assert main(["evaluate","--prompt","prompts/banking77_v1.yaml","--set-baseline",
                 "--export-baseline",str(baseline),*common])==0
    assert main(["compare","--prompt","prompts/banking77_v1.yaml",*common])==0
    assert main(["compare","--prompt","prompts/banking77_v2.yaml","--baseline-file",str(baseline),*common])==1
    rules=yaml.safe_load((ROOT/"config/thresholds.yaml").read_text())
    for metric in ("accuracy","macro_f1"):
        rules[metric]={"warning_drop":.001,"critical_drop":1}
    warn=tmp_path/"warn.yaml"
    warn.write_text(yaml.safe_dump(rules))
    assert main(["compare","--prompt","prompts/banking77_v2.yaml","--thresholds",str(warn),*common])==0
    assert main(["compare","--prompt","prompts/banking77_v2.yaml","--baseline-run-id","missing",*common])==2
    assert main(["evaluate","--prompt","prompts/banking77_v1.yaml","--concurrency","0",*common])==2
    assert len(list(tmp_path.glob("*.html")))==4
    summary=(tmp_path/"summary.md").read_text()
    assert "FINAL STATUS: FAIL" in summary and "FINAL STATUS: WARN" in summary
    assert "macro_f1" in summary and "improvements" in summary
    db=Database(tmp_path/"demo.sqlite")
    assert db.connection.execute("SELECT COUNT(*) FROM judge_results").fetchone()[0]>0
    db.close()


def test_baseline_mismatch_preflight_blocks_provider(tmp_path,monkeypatch):
    monkeypatch.chdir(ROOT)
    common=["--offline","--db",str(tmp_path/"db.sqlite"),"--output-dir",str(tmp_path)]
    assert main(["evaluate","--prompt","prompts/banking77_v1.yaml","--set-baseline",*common])==0
    def forbidden(*args):
        pytest.fail("Provider must not be instantiated for an incompatible baseline")
    monkeypatch.setattr("llmguard.cli.create_client",forbidden)
    assert main(["compare","--prompt","prompts/banking77_v2.yaml","--seed","7",*common])==2

