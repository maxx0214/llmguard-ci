# LLMGuard CI

LLMGuard evaluates prompt versions against the official BANKING77 intent taxonomy and gates regressions with deterministic metrics, selective semantic judging, SQLite persistence, and an HTML report.

## Install

```powershell
python -m pip install -e .
python -m pytest -q
```

The only live provider is OpenAI. Configure `OPENAI_API_KEY`, `OPENAI_CLASSIFIER_MODEL`, and optionally `OPENAI_JUDGE_MODEL` (the latter falls back to the classifier model). No other provider SDK or credential is used.

## Dataset and sampling

The loader fetches the official PolyAI CSVs at a pinned immutable revision through Hugging Face `datasets`; no examples are vendored. `smoke` (100) and `regression` (500) use seeded, deterministic stratified round-robin sampling: every intent is represented in smoke, and regression is a superset for the same seed. `full` evaluates the complete official split (3,080 test cases). Set `HF_DATASETS_OFFLINE=1` to require an existing cache.

On Windows, unusually long cache roots are mapped to a stable short hashed directory under the system temporary directory so `datasets` lock filenames remain valid.

## Commands

```powershell
python -m llmguard.cli evaluate --prompt prompts/banking77_v1.yaml --tier smoke --set-baseline --export-baseline baselines/smoke.json
python -m llmguard.cli compare --prompt prompts/banking77_v2.yaml --tier smoke --baseline-file baselines/smoke.json
python -m llmguard.cli evaluate --prompt prompts/banking77_v1.yaml --offline --set-baseline --db results/demo.sqlite --output-dir results/demo
python -m llmguard.cli compare --prompt prompts/banking77_v2.yaml --offline --db results/demo.sqlite --output-dir results/demo
```

`--offline` uses 100 explicit synthetic fixtures and makes no download or API call. Exit code is `0` for PASS/WARN, `1` for a quality-gate FAIL, and `2` for configuration or execution errors.

## Reports and storage

Each run writes JSON and an escaped responsive HTML report under `results/`. SQLite stores immutable run snapshots, case results, and judge results. Accuracy and Macro F1 are computed over all attempted cases; the confusion matrix includes an explicit error column.

## CI

Pull requests run tests and a smoke comparison. Manual workflow dispatch can select `smoke`, `regression`, or the paid `full` tier. The workflow reads `OPENAI_API_KEY` from a GitHub secret and model names from repository variables.
