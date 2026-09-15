"""Self-contained Jinja2 reports and consistent terminal/GitHub summaries."""
from pathlib import Path
from jinja2 import Environment, PackageLoader, select_autoescape
from llmguard.models import EvaluationRun, RegressionResult

PRIMARY = ("accuracy", "macro_f1", "weighted_f1", "latency_p95", "estimated_cost")


def format_metric(name: str, value: float | None, delta: bool = False) -> str:
    if value is None:
        return "N/A"
    sign = "+" if delta else ""
    if name in ("accuracy", "macro_f1", "weighted_f1"):
        return f"{value * 100:{sign}.2f}" + (" pp" if delta else "%")
    if name.startswith("latency"):
        return f"{value:{sign}.1f} ms"
    if "cost" in name:
        return f"{value:{sign}.6f} USD"
    return f"{value:{sign}.2f}"


def write_report(run: EvaluationRun, path: Path, baseline: EvaluationRun | None = None,
                 regression: RegressionResult | None = None) -> None:
    env = Environment(loader=PackageLoader("llmguard.report", "templates"),
                      autoescape=select_autoescape(["html"]))
    env.globals["format_metric"] = format_metric
    html = env.get_template("report.html").render(
        run=run, baseline=baseline, regression=regression,
        previous={c.test_case_id: c for c in baseline.cases} if baseline else {},
        cases={c.test_case_id: c for c in run.cases},
        regressed=set(regression.transitions["regressions"]) if regression else set(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def run_label(run: EvaluationRun) -> str:
    return f"{run.prompt.id} v{run.prompt.version} / {run.prompt.provider} / {run.prompt.model}"


def markdown_summary(run: EvaluationRun, baseline: EvaluationRun | None,
                     regression: RegressionResult | None) -> str:
    lines = ["## LLMGuard CI - BANKING77", "", f"**FINAL STATUS: {run.status}**", "",
             f"Dataset: {run.dataset.dataset} / {run.dataset.split} / {run.dataset.tier} / "
             f"{run.dataset.sample_count} cases / seed {run.dataset.seed}", "",
             f"Baseline: {run_label(baseline) if baseline else 'none'}",
             f"Candidate: {run_label(run)}", ""]
    if run.metadata.get("mode") == "offline":
        lines += ["OFFLINE SYNTHETIC FIXTURES: not benchmark/model performance.", ""]
    lines += ["| Metric | Baseline | Candidate | Delta |", "|---|---:|---:|---:|"]
    for name in PRIMARY:
        value = getattr(run.metrics, name)
        before = getattr(baseline.metrics, name) if baseline else None
        delta = value-before if value is not None and before is not None else None
        lines.append(f"| {name} | {format_metric(name, before)} | {format_metric(name, value)} | "
                     f"{format_metric(name, delta, True)} |")
    if regression:
        for kind, ids in regression.transitions.items():
            lines.append(f"\n{kind}: {len(ids)}" + (f" ({', '.join(ids)})" if kind != "stable_successes" and ids else ""))
    m = run.metrics
    lines += ["", f"Judge: {m.judge_count} analyzed / {m.judge_error_count} errors; severity: {m.judge_severity_counts}",
              f"Classifier tokens: {m.input_tokens} input / {m.output_tokens} output",
              f"Judge tokens: {m.judge_input_tokens} input / {m.judge_output_tokens} output",
              f"Judge cost: {format_metric('cost', m.judge_cost)}; total: {format_metric('cost', m.total_estimated_cost)}",
              f"Invalid outputs: {m.invalid_predictions}; execution errors: {m.error_count}", ""]
    return "\n".join(lines)


def terminal_summary(run: EvaluationRun, baseline: EvaluationRun | None,
                     regression: RegressionResult | None) -> str:
    lines = ["LLMGuard CI - BANKING77",
             f"Dataset: {run.dataset.dataset} {run.dataset.split} / {run.dataset.tier} / {run.dataset.sample_count} cases / seed {run.dataset.seed}",
             f"Candidate: {run_label(run)}"]
    if run.metadata.get("mode") == "offline":
        lines.append("OFFLINE SYNTHETIC FIXTURES: no real benchmark/model performance.")
    if baseline:
        lines.append(f"Baseline: {run_label(baseline)}")
    lines.append("")
    for name in PRIMARY:
        value = getattr(run.metrics, name)
        before = getattr(baseline.metrics, name) if baseline else None
        delta = value-before if value is not None and before is not None else None
        lines.append(f"{name:18} {format_metric(name,before):>14} -> {format_metric(name,value):>14}"
                     + (f" ({format_metric(name,delta,True)})" if baseline else ""))
    if regression:
        lines += ["", *regression.reasons]
        lines.append(" / ".join(f"{kind}: {len(ids)}" for kind, ids in regression.transitions.items()))
    lines += [f"Judge severity: {run.metrics.judge_severity_counts}; errors: {run.metrics.judge_error_count}",
              f"Invalid outputs: {run.metrics.invalid_predictions}; execution errors: {run.metrics.error_count}"]
    return "\n".join(lines)
