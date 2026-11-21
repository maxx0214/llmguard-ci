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

## Generic evaluation suites (experimental)

The original BANKING77 commands and artifacts are unchanged. The new
`suite` command accepts task-specific user data in JSON, JSONL or CSV and either
precomputed predictions or an explicitly configured chat-completions provider.

| Task | Ground truth (`expected`) | Prediction (`prediction`) | Metrics |
| --- | --- | --- | --- |
| classification | label string | label string | Accuracy, macro F1 |
| qa | reference answer string | response string | Normalized exact match, whitespace-token F1 |
| retrieval | unique relevant document IDs | unique ranked document IDs (possibly empty) | Hit@K, Recall@K, MRR@K, binary nDCG@K |

Every dataset row needs a unique string `id`, a nonempty `input` and `expected`.
Prediction rows contain `id` and `prediction`; IDs must match the dataset.
For CSV retrieval, encode list fields as JSON arrays inside quoted cells.
Missing or invalid predictions count toward errors and fail the gate.
Classification labels are inferred from the dataset. QA matching is lexical,
not a semantic judge; retrieval scoring uses binary relevance.

### Offline quick start

```bash
llmguard suite --config examples/suite/retrieval.yaml --output results/retrieval-baseline.json
llmguard suite --config examples/suite/retrieval-candidate.yaml --output results/retrieval-candidate.json --baseline results/retrieval-baseline.json
```

The candidate intentionally regresses. Both JSON and HTML are written; the
comparison returns exit code 1 on a regression or prediction error, and 2
on invalid configuration. The baseline must be error-free and match the
task, dataset fingerprint and retrieval depth. Per-metric tolerances are
absolute score drops (0 to 1); unspecified tolerances are zero.

### Live model evaluation

For classification and QA, replace `predictions` in the suite YAML with a
provider configuration:

```yaml
task: qa
dataset: questions.jsonl
provider:
  name: openai
  model: YOUR_MODEL
  api_key_env: OPENAI_API_KEY
  system_prompt: Answer directly and concisely.
tolerances:
  exact_match: 0.02
  token_f1: 0.02
```

The provider adapter uses the OpenAI SDK's chat-completions interface.
Other services serving that compatible API (including local model servers)
can be selected by setting a distinct provider name, model, credential
environment variable, and explicit `base_url`. Compatibility varies by
service/model; no native non-OpenAI SDK or live provider verification is included.
Model requests send dataset inputs to the selected endpoint and may incur fees.
Never put credentials in YAML or commit sensitive data or generated reports.

For retrieval, export ranked document IDs from your own retrieval pipeline
and score them locally; no LLM provider is needed. The suite's JSON and HTML
are separate from the legacy SQLite/BANKING77 baseline format.
