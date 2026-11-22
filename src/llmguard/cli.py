"""PASS/WARN=0, gate FAIL=1, execution/configuration error=2."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmguard.config import Settings
    from llmguard.llm.client import LLMClient
    from llmguard.models import EvaluationRun, PromptConfig


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="llmguard", description="LLMGuard CI: behavioral regression testing (use llmguard suite --help for generic suites)")
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "compare"):
        sub = commands.add_parser(command, help=f"{command.capitalize()} a versioned BANKING77 prompt")
        sub.add_argument("--prompt", type=Path, required=True)
        sub.add_argument("--tier", choices=("smoke", "regression", "full"), default="smoke")
        sub.add_argument("--split", choices=("test", "train"), default="test")
        sub.add_argument("--seed", type=int)
        sub.add_argument("--config", type=Path, default=Path("config/evaluation.yaml"))
        sub.add_argument("--db", type=Path, default=Path("results/banking77.sqlite"))
        sub.add_argument("--output-dir", type=Path, default=Path("results"))
        sub.add_argument("--cache-dir", type=Path, default=Path(".cache/datasets"))
        sub.add_argument("--pricing", type=Path, default=Path("config/pricing.yaml"))
        sub.add_argument("--thresholds", type=Path, default=Path("config/thresholds.yaml"))
        sub.add_argument("--concurrency", type=int)
        sub.add_argument("--timeout", type=float)
        sub.add_argument("--model", help="Classifier model override")
        sub.add_argument("--judge-model")
        sub.add_argument("--disable-judge", action="store_true")
        sub.add_argument("--judge-max-cases", type=int)
        sub.add_argument("--offline", action="store_true", help="100 synthetic fixtures; no download, API or credential")
        if command == "evaluate":
            sub.add_argument("--set-baseline", action="store_true")
            sub.add_argument("--export-baseline", type=Path)
        else:
            group = sub.add_mutually_exclusive_group()
            group.add_argument("--baseline-run-id")
            group.add_argument("--baseline-file", type=Path)
    return root


def create_client(prompt: PromptConfig, timeout: float, offline: bool, cases: list) -> LLMClient:
    if offline:
        from llmguard.llm.offline import OfflineClient
        return OfflineClient(cases)
    from llmguard.llm.openai import OpenAIClient
    return OpenAIClient(timeout)


def effective_settings(args: argparse.Namespace) -> Settings:
    from llmguard.config import Settings, load_yaml

    data = load_yaml(args.config)
    settings = Settings.model_validate(data)
    values = settings.evaluation.model_dump()
    for key, env in (("concurrency", "LLMGUARD_CONCURRENCY"), ("timeout", "LLMGUARD_TIMEOUT")):
        raw = getattr(args, key)
        if raw is None:
            raw = os.getenv(env)
        if raw is not None:
            values[key] = raw
    if args.seed is not None:
        values["seed"] = args.seed
    judges = settings.judge.model_dump()
    if args.disable_judge:
        judges["enabled"] = False
    if args.judge_max_cases is not None:
        judges["max_cases"] = args.judge_max_cases
    return Settings.model_validate({"evaluation": values, "judge": judges})


def write_artifacts(run: EvaluationRun, baseline: EvaluationRun | None, regression,
                    directory: Path) -> Path:
    from llmguard.report.html_report import markdown_summary, write_report

    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / run.run_id
    stem.with_suffix(".json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    write_report(run, stem.with_suffix(".html"), baseline, regression)
    summary = markdown_summary(run, baseline, regression)
    stem.with_suffix(".md").write_text(summary, encoding="utf-8")
    if regression:
        (directory / f"{run.run_id}.regression.json").write_text(regression.model_dump_json(indent=2), encoding="utf-8")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
    return stem.with_suffix(".html")


async def execute(args: argparse.Namespace) -> int:
    from llmguard.config import PricingConfig, load_prompt, load_yaml
    from llmguard.eval.metrics import compute_metrics
    from llmguard.eval.runner import evaluate, run_metadata
    from llmguard.llm.offline import fixture_dataset
    from llmguard.models import EvaluationRun
    from llmguard.regression.comparator import compare, ensure_compatible
    from llmguard.regression.thresholds import load_thresholds
    from llmguard.report.html_report import terminal_summary
    from llmguard.storage.database import Database, export_baseline, read_baseline

    settings = effective_settings(args)
    config, judge = settings.evaluation, settings.judge
    prompt = load_prompt(args.prompt, args.model or ("offline-fixture" if args.offline else None))
    judge_model = (args.judge_model or os.getenv(f"{prompt.provider.upper()}_JUDGE_MODEL")
                   or judge.model or str(prompt.model))
    pricing = PricingConfig.model_validate(load_yaml(args.pricing))
    thresholds = load_thresholds(args.thresholds)
    if thresholds.judge_semantic_fit.enabled_as_gate and (not judge.enabled or judge.max_cases == 0):
        raise ValueError("Judge gate requires enabled judging and a positive max_cases")
    if args.command == "evaluate" and args.export_baseline and not args.set_baseline:
        raise ValueError("--export-baseline requires --set-baseline")
    if args.offline:
        if args.tier != "smoke" or args.split != "test":
            raise ValueError("--offline uses synthetic smoke fixtures only")
        cases, dataset = fixture_dataset()
        dataset.seed = config.seed
    else:
        from llmguard.data.banking77 import load_banking77
        cases, dataset = load_banking77(args.tier, config.seed, args.split, config, args.cache_dir)
    db = Database(args.db)
    try:
        baseline = None
        if args.command == "compare":
            baseline = (read_baseline(args.baseline_file) if args.baseline_file else
                        db.baseline(prompt.id, dataset, "offline" if args.offline else "live", args.baseline_run_id))
            preflight = EvaluationRun(prompt=prompt, dataset=dataset, cases=[], metrics=compute_metrics([]),
                                      metadata=run_metadata(prompt, pricing, judge_model, judge, args.offline, config.concurrency))
            ensure_compatible(baseline, preflight)
        client = create_client(prompt, config.timeout, args.offline, cases)
        try:
            run = await evaluate(prompt, cases, dataset, client, pricing, judge_model, judge,
                                 config.concurrency, baseline, args.offline)
        finally:
            await client.close()
        regression = compare(baseline, run, thresholds) if baseline else None
        if regression:
            run.status = regression.status
        errors = run.metrics.error_count + run.metrics.invalid_predictions
        run.is_baseline = bool(args.command == "evaluate" and args.set_baseline and errors == 0)
        run.metadata["thresholds"] = thresholds.model_dump()
        db.save(run)
        report = write_artifacts(run, baseline, regression, args.output_dir)
        if args.command == "evaluate" and args.export_baseline and run.is_baseline:
            export_baseline(run, args.export_baseline)
        print(terminal_summary(run, baseline, regression))
        print(f"Run: {run.run_id}\nReport: {report}\nFINAL STATUS: {run.status}")
        return 1 if run.status == "FAIL" else 0
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        pass
    else:
        load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "suite":
        from llmguard.suite import main as suite_main
        return suite_main(argv[1:])
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        return asyncio.run(execute(args))
    except Exception as exc:
        # SDK response bodies and tracebacks can contain user data or credentials.
        if isinstance(exc, ModuleNotFoundError) and not (exc.name or "").startswith("llmguard"):
            message = "BANKING77 requires pip install 'llmguard-ci[banking77]'"
        else:
            message = str(exc) if isinstance(exc, (ValueError, OSError)) else type(exc).__name__
        print(f"LLMGuard execution/configuration error: {message}", file=sys.stderr)
        if os.getenv("GITHUB_STEP_SUMMARY"):
            try:
                with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as handle:
                    handle.write("## LLMGuard CI\n\n**EXECUTION ERROR (exit 2)**. See job logs.\n")
            except OSError:
                pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

