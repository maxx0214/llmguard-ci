"""Transactional SQLite v2; legacy databases are rejected without mutation."""
import sqlite3
from pathlib import Path

from llmguard.config import fingerprint
from llmguard.eval.metrics import compute_metrics
from llmguard.models import DatasetInfo, EvaluationRun

SCHEMA_VERSION = 2
SCHEMA = """
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, dataset_split TEXT NOT NULL,
    evaluation_tier TEXT NOT NULL, sample_count INTEGER NOT NULL, seed INTEGER NOT NULL,
    dataset_fingerprint TEXT NOT NULL, prompt_id TEXT NOT NULL, prompt_version INTEGER NOT NULL,
    provider TEXT NOT NULL, model TEXT NOT NULL, timestamp TEXT NOT NULL,
    accuracy REAL NOT NULL, macro_precision REAL NOT NULL, macro_recall REAL NOT NULL,
    macro_f1 REAL NOT NULL, weighted_f1 REAL NOT NULL,
    latency_mean REAL NOT NULL, latency_p50 REAL NOT NULL, latency_p95 REAL NOT NULL,
    input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, estimated_cost REAL,
    valid_predictions INTEGER NOT NULL, invalid_predictions INTEGER NOT NULL,
    error_count INTEGER NOT NULL, status TEXT NOT NULL, is_baseline INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX baseline_lookup ON runs(prompt_id, is_baseline, dataset_fingerprint, timestamp DESC);
CREATE TABLE case_results (
    run_id TEXT NOT NULL REFERENCES runs(run_id), test_case_id TEXT NOT NULL,
    input_text TEXT NOT NULL, expected_intent TEXT NOT NULL, predicted_intent TEXT,
    correct INTEGER NOT NULL, latency_ms REAL NOT NULL, input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL, estimated_cost REAL, error TEXT, error_kind TEXT,
    payload TEXT NOT NULL, PRIMARY KEY(run_id,test_case_id)
);
CREATE TABLE judge_results (
    run_id TEXT NOT NULL, test_case_id TEXT NOT NULL,
    semantic_fit INTEGER, severity TEXT, ambiguity TEXT, error_type TEXT,
    explanation TEXT, error TEXT, latency_ms REAL NOT NULL,
    input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, estimated_cost REAL,
    payload TEXT NOT NULL, PRIMARY KEY(run_id,test_case_id),
    FOREIGN KEY(run_id,test_case_id) REFERENCES case_results(run_id,test_case_id)
);
"""


def validate_snapshot(run: EvaluationRun) -> None:
    if run.dataset.sample_count != len(run.cases) or len({c.test_case_id for c in run.cases}) != len(run.cases):
        raise ValueError("Run case count or IDs are inconsistent")
    cases = [{"id": c.test_case_id, "text": c.input_text, "expected_intent": c.expected_intent} for c in run.cases]
    if fingerprint(cases) != run.dataset.fingerprint:
        raise ValueError("Run dataset fingerprint does not match its cases")
    if len({j.test_case_id for j in run.judgments}) != len(run.judgments):
        raise ValueError("Duplicate judge results")
    if not {j.test_case_id for j in run.judgments} <= {c.test_case_id for c in run.cases}:
        raise ValueError("Judge results reference unknown cases")
    if compute_metrics(run.cases, run.judgments) != run.metrics:
        raise ValueError("Run metrics do not match case evidence")


def require_baseline(run: EvaluationRun) -> None:
    validate_snapshot(run)
    if not run.is_baseline or run.metrics.error_count or run.metrics.invalid_predictions:
        raise ValueError("An error-free designated baseline is required")


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        try:
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            tables = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if version == 0 and not tables:
                self.connection.executescript(f"BEGIN;\n{SCHEMA}\nPRAGMA user_version=2;\nCOMMIT;")
            elif version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported/legacy SQLite schema {version}. Database left intact; use a new --db path for BANKING77.")
            expected = {"runs", "case_results", "judge_results"}
            actual = {r[0] for r in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not expected <= actual:
                raise ValueError("Incomplete SQLite v2 schema")
            self.connection.execute("PRAGMA foreign_keys=ON")
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def _insert(self, table: str, data: dict) -> None:
        columns = ",".join(data)
        marks = ",".join("?" for _ in data)
        self.connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(data.values()))

    def save(self, run: EvaluationRun) -> None:
        validate_snapshot(run)
        m, d = run.metrics, run.dataset
        with self.connection:
            self._insert("runs", {
                "run_id": run.run_id, "dataset": d.dataset, "dataset_split": d.split,
                "evaluation_tier": d.tier, "sample_count": d.sample_count, "seed": d.seed,
                "dataset_fingerprint": d.fingerprint, "prompt_id": run.prompt.id,
                "prompt_version": run.prompt.version, "provider": run.prompt.provider,
                "model": run.prompt.model, "timestamp": run.timestamp.isoformat(),
                **{k: getattr(m, k) for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "weighted_f1", "latency_mean", "latency_p50", "latency_p95", "input_tokens",
                    "output_tokens", "estimated_cost", "valid_predictions", "invalid_predictions", "error_count")},
                "status": run.status, "is_baseline": int(run.is_baseline), "payload": run.model_dump_json()})
            for case in run.cases:
                data = case.model_dump(exclude={"usage"})
                self._insert("case_results", {"run_id": run.run_id, **data, "payload": case.model_dump_json()})
            for judgment in run.judgments:
                result = judgment.result.model_dump() if judgment.result else dict.fromkeys(
                    ("semantic_fit", "severity", "ambiguity", "error_type", "explanation"))
                self._insert("judge_results", {"run_id": run.run_id, "test_case_id": judgment.test_case_id,
                    **result, "error": judgment.error, "latency_ms": judgment.latency_ms,
                    "input_tokens": sum(u.input_tokens for u in judgment.usage),
                    "output_tokens": sum(u.output_tokens for u in judgment.usage),
                    "estimated_cost": judgment.estimated_cost, "payload": judgment.model_dump_json()})

    def baseline(self, prompt_id: str, dataset: DatasetInfo, mode: str,
                 run_id: str | None = None) -> EvaluationRun:
        if run_id:
            rows = self.connection.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,))
        else:
            rows = self.connection.execute(
                "SELECT payload FROM runs WHERE prompt_id=? AND is_baseline=1 ORDER BY timestamp DESC, rowid DESC",
                (prompt_id,))
        for (payload,) in rows:
            run = EvaluationRun.model_validate_json(payload)
            if run_id or (run.dataset == dataset and run.metadata.get("mode") == mode):
                require_baseline(run)
                if run.prompt.id != prompt_id:
                    raise ValueError("Baseline prompt ID does not match")
                return run
        raise ValueError("No compatible baseline. Run evaluate --set-baseline or supply --baseline-file.")


def read_baseline(path: Path) -> EvaluationRun:
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("Legacy baseline JSON is not compatible with BANKING77; generate a new baseline")
    run = EvaluationRun.model_validate(payload)
    require_baseline(run)
    return run


def export_baseline(run: EvaluationRun, path: Path) -> None:
    require_baseline(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

