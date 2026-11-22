# Usage guide

## Data handling

The generic SDK omits input, expected values, predictions, and contexts from
reports unless `include_payloads=True`. Case IDs, caller-supplied version
labels, custom metadata, and measurement counters remain in reports. Use
opaque IDs and version labels and pass only metadata you intend to share.
Generic suite and BANKING77 artifacts include case payloads.

SQLite caches and checkpoints store normalized predictions even with report
payloads disabled. Restrict filesystem access and retention for state files
and their backups. HTTP adapters send mapped input to the selected service.
The built-in REST and OpenAI paths suppress their request debug logs. Custom
callables, adapters, and metric callbacks control their own logging.
REST state is scoped to the configured authentication headers. A credential
change starts a new cache scope and requires a new checkpoint. Upgrade from
the pre-release REST state format by using fresh files. For local callables,
change the model version, prompt version, or system settings when external
state affecting predictions changes.

## Python SDK

Pass a JSON, JSONL, or CSV file or a list of case dictionaries. Each case has a
unique string `id`, an `input`, and an `expected` value. The SDK passes `input`
as the callable's only argument. JSON datasets support nonempty objects and
arrays as well as strings. Use `qa`, `classification`, or `retrieval` with the
ground-truth formats and profiles documented below.

```python
from llmguard import Evaluator, System

evaluator = Evaluator("test.jsonl", "retrieval", ["recall@5", "mrr@5", "ndcg@5"])
baseline = evaluator.evaluate(System(rag_v1, model_version="rag-v1", prompt_version="prompt-v1"))
baseline.report("results/baseline.json")
candidate = evaluator.compare(baseline="results/baseline.json", candidate=rag_v2)
candidate.report("results/candidate.json")
```

`evaluate()` and `compare()` accept synchronous or asynchronous callables.
Inside an existing event loop, await `aevaluate()` or `acompare()`. Cases run
sequentially by default. The evaluator runs blocking callables in worker threads.
`result.report()` returns a dictionary. Adding a `.json` path also saves JSON
and a sibling HTML report. `result.status`, `result.metrics`, and `result.cases`
provide direct access to the verdict, scores, and case records.

The HTML from `compare()` is self-contained. Metrics and Operations show
Baseline and Candidate values, their difference, and the gate side by side.
The same page shows both runs' execution details and matches case results by
ID. Paired statistics appear when enabled.

A saved SDK baseline must be error-free and match the dataset fingerprint,
task, profile, metric selection, and metric versions. Model and prompt
versions are supplied through `System` and recorded when provided. Reports
also record evaluation settings, the Python and OS environment, per-case
timing and errors, and the metric contract. Case input, gold values,
predictions, and contexts are omitted by default. Set `include_payloads=True`
to record those mapped values in a private report. Use opaque case IDs and
version labels when reports will be shared.

### Saved predictions and dataset mapping

```python
from llmguard import Evaluator, PredictionFile

evaluator = Evaluator(
    "testset.json", "retrieval", ["recall@5", "mrr@5", "ndcg@5"],
    dataset_mapping={"id": "id", "input": "question", "expected": "gold_chunks.*.chunk_id"},
)
result = evaluator.compare(
    baseline=PredictionFile("hybrid.json"),
    candidate=PredictionFile("reranked.json", {"retrieved_ids": "ranked_document_ids"}),
)
result.report("results/comparison.json")
```

The dataset in this example contains rows like
`{"id":"q1","question":"Where is the return policy?","gold_chunks":[{"chunk_id":"returns"}]}`.
The baseline prediction file contains rows like `{"id":"q1","retrieved_ids":["returns"]}`.
The candidate uses `ranked_document_ids` for the ranked IDs. Use `rows_path="data.items"`
for predictions nested in a JSON envelope, `dataset_rows_path` for an enveloped
dataset, and `id_path` for a custom prediction ID field. Both mappings use the
same dot paths, array indices, and wildcards as output mapping. Equivalent
canonical and mapped datasets have the same fingerprint.

Each prediction file must have exactly one row for each dataset ID. Record
order may differ. Duplicate, missing, or unexpected IDs fail preflight.
Use `answer`, `label`, or `retrieved_ids` for task-specific output. The
`prediction` field and mapping key also work. CSV cells are strings. Encode
retrieval rankings as JSON arrays inside quoted cells.

`PredictionFile` works with `evaluate()`, `compare()`, and their async
variants. Either comparison side can be a file, a callable, or a REST
adapter. A plain path passed as a baseline still means a saved SDK report.
Replays record zero requests and zero actual API charges. Historical
per-case latency, usage, and cost are unknown unless supplied in the
predictions. Model version defaults to the file stem and can be set explicitly.

### Output mapping and model examples

Strings map to `answer` for QA or `label` for classification. A list of IDs
maps to `retrieved_ids` for retrieval. Dictionaries and Python objects can
expose those same field names directly. The evaluator also accepts the
optional `contexts`, `latency_ms`, `usage`, `cost`, and `metadata` fields.

Use `output_mapping` on the evaluator or on a `System` to project fields from
another format. Dot-separated paths traverse dictionaries and object
attributes. A numeric path segment selects a list index, and `*` selects each
list element. The `System` mapping replaces the evaluator's default mapping
for that system. A custom `adapter` can transform an output before mapping,
including an async transformation. Missing fields required for the selected
metrics produce explicit case errors.

```python
def local_llm(question):
    return local_chat_model(question)  # Returns answer text.

qa = Evaluator("questions.jsonl", "qa", ["exact_match", "token_f1"])
qa.evaluate(local_llm)

# A RAG pipeline can return nested dictionaries or objects.
rag = System(my_rag, output_mapping={
    "retrieved_ids": "search.hits.*.document_id",
    "contexts": "search.hits.*.text",
})
evaluator.evaluate(rag)

# A LangChain retriever returns Document objects. Use your own stable IDs.
langchain_rag = System(retriever.invoke, output_mapping={
    "retrieved_ids": "*.metadata.doc_id",
    "contexts": "*.page_content",
})
evaluator.evaluate(langchain_rag)
```

Wrap a fine-tuned decoder-only model's `generate()` method to encode prompts
and decode the new tokens. The tokenizer and model are loaded by the application.

```python
def generate_answer(prompt):
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**encoded, max_new_tokens=64)
    completion = output_ids[0, encoded["input_ids"].shape[1]:]
    return {"answer": tokenizer.decode(completion, skip_special_tokens=True)}

qa.evaluate(System(generate_answer, model_version="checkpoint-42"))
```

### Case metadata and custom metrics

```python
from llmguard import Evaluator, Metric, System

evaluator = Evaluator(
    "test.jsonl", "retrieval", ["recall@5", "fallback_rate"],
    metadata_fields=["rerank_failed"],
    custom_metrics={
        "fallback_rate": Metric(
            lambda case: case["metadata"].get("rerank_failed"),
            direction="lower_is_better", version="1",
        ),
    },
    quality_gate={"max_scores": {"fallback_rate": 0.05},
                  "max_drop": {"fallback_rate": 0.02}, "on_unknown": "warn"},
)
rag = System(my_rag, output_mapping={
    "retrieved_ids": "search.hits.*.document_id",
    "metadata.rerank_failed": "diagnostics.rerank_failed",
})
evaluator.evaluate(rag).report("results/fallback.json")
```

Return a `metadata` dictionary directly or map individual keys as
`metadata.<name>` from live or archived output. Reports retain these fields
even when payloads are off. Use non-sensitive values. The report counts
known and unknown values and shows the frequency of each known value.

A custom metric receives a case with `id`, `input`, `expected`, the
normalized `prediction`, `metrics`, `error`, and `metadata`. Return a finite
number or boolean for each case, or `None` if the value is unknown. The
aggregate is the arithmetic mean when every case has a value. A bare callable
defaults to higher is better. Set `min_scores` for a higher-is-better metric,
`max_scores` for a lower-is-better metric, and `max_drop` for an allowed
degradation in either direction. Increment the metric's version when its
scoring changes. Paired hypothesis tests cover built-in quality metrics.

### Quality gates

```python
policy = {
    "min_scores": {"recall@5": 0.80},
    "max_drop": {"recall@5": {"warn": 0.01, "fail": 0.03}, "latency_ms": 50},
    "max_latency_ms": 800,
    "max_cost": 0.05,
    "max_error_rate": 0.02,
    "on_unknown": "fail",
}
evaluator = Evaluator("test.jsonl", "retrieval", ["recall@5"], quality_gate=policy)
```

A number sets a failure bound. A mapping can set a warning bound, a failure
bound, or both. A value exactly at its bound is allowed. Quality regressions
default to zero tolerance and the default maximum error rate is zero.
`max_drop` is an absolute allowed degradation. Higher quality scores are
better, while lower latency, cost, and error rates are better. The latency
limit uses the slowest measured request, including retries. A separately
reported latency is available per case. Cost is the summed explicit USD value
from each response. Usage is a set of caller-reported numeric counters.
Missing coverage makes totals unknown, represented as JSON `null`. A gated
unknown defaults to `FAIL` and can be set to `WARN` with `on_unknown`.

### Parallel execution, cache, and resume

```python
from llmguard import Evaluator, ExecutionOptions, System

execution = ExecutionOptions(
    max_concurrency=8, requests_per_second=10, timeout=5,
    retries=2, retry_backoff=0.2, max_retry_delay=30,
)
evaluator = Evaluator("test.jsonl", "retrieval", ["ndcg@10"],
                      execution=execution, cache="results/cache.sqlite")
system = System(rag_v2, model_version="my-rag@v2", prompt_version="search-v3",
                settings={"temperature": 0, "index_revision": "2026-11"})
result = evaluator.evaluate(system, checkpoint="results/rag-v2.sqlite")
result.report("results/rag-v2.json")

# Reuse completed successes and rerun any failed or unfinished cases.
evaluator.evaluate(system, checkpoint="results/rag-v2.sqlite", retry_failed=True)
```

`requests_per_second` spaces request starts for each evaluation and applies to
retries. The SDK retries connection failures, timeouts, and transient HTTP
statuses. Python systems can raise `RateLimitError(retry_after=...)` to signal
a shared cooldown. REST also honors the server's `Retry-After` on transient
responses. A requested wait above `max_retry_delay` stops new requests in that
evaluation. If the SDK retry count is omitted, REST uses its adapter setting
and Python callables get one attempt. The SDK timeout applies per attempt.
Python cannot forcibly stop a running thread, so a timed-out synchronous call
holds its slot until it returns. Reports preserve the canonical case-ID order.

The SQLite cache is opt-in. It keys normalized successful responses by input,
model and prompt versions, `System.settings`, output mapping, and the execution
and scoring contract. Change the versions or settings when a model, prompt,
index, service account scope, or custom adapter changes. Use `prompt_version="none"`
for systems without a prompt. Use `cache=None` or set
`ExecutionOptions(independent_trials=True)` for independent stochastic trials.
The latter disables cache reads and writes even when a cache path is supplied.
Use a fresh checkpoint path for each independent trial.

Checkpoints save each finished case in a transaction. Rerun with the same
checkpoint path and dataset, system, and evaluation settings to resume. A
checkpoint has one active owner and uses a `.lease` sidecar. For comparison,
pass `checkpoint=` for the candidate and call `evaluate()` separately if the
baseline needs its own checkpoint. A request interrupted before its result is
committed may run again. REST services that have side effects should handle
duplicate requests. Cache and checkpoint databases can contain normalized
answers or document IDs, so store them privately. Avoid putting secrets in
`System.settings` and provide auth through the REST adapter's environment
variables.

JSON and HTML distinguish recorded service measurements from this run's wall
time, cache hits, resumed cases, request attempts, API calls, usage, and cost.
REST counts its HTTP attempts. Set `System(is_api=True)` for a callable that
makes one API attempt per invocation. The saved-call estimate counts one
avoided API invocation per backed cache hit. When a retry's cost or usage is
unobserved, the corresponding execution totals remain unknown.

### Paired statistical regression

Install the `statistics` extra for paired comparisons.

```python
from llmguard import Evaluator, StatisticsOptions

evaluator = Evaluator(
    "test.jsonl", "retrieval", ["recall@5", "mrr@5", "ndcg@5"],
    statistics=StatisticsOptions(seed=42, min_effect={"ndcg@5": 0.02},
                                 correction="holm", enforce=False),
)
result = evaluator.compare("results/baseline.json", rag_v2)
result.report("results/statistical-candidate.json")
```

`statistics=True` uses the defaults of 2,000 bootstrap draws, 2,000
permutations, 95% intervals, `alpha=.05`, at least 20 pairs, and an absolute
practical drop of .01. The reports pair by case ID and show candidate minus
baseline effects, paired percentile bootstrap intervals, two-sided p-values,
and adjusted p-values. Accuracy and Exact Match use exact McNemar tests.
Other scores use paired permutation tests with exact enumeration when the
swap space fits the draw budget. Macro F1 is recomputed over each complete
resampled or permuted classification sample using the same suite scorer.
New SDK reports record class indices for replay. Older redacted baselines can
still be compared, with Macro F1 inference marked unavailable when the
predictions needed to recompute it are missing.

Holm adjusts the p-values across the selected quality metrics. Bonferroni
and uncorrected comparisons are also available. Confidence intervals are
pointwise. A statistical regression requires both a significant adjusted
p-value and a drop reaching the practical threshold. Insufficient pairs are
reported as uncertain. `enforce=True` lets a statistical regression fail the
run in addition to the existing threshold gates. `enforce=False` reports
evidence while leaving the threshold verdict in control. To replay saved
artifacts directly, pass their JSON dictionaries to `paired_statistics`.

Inference assumes independent cases and exchangeability within pairs under
the null. Questions sharing passages or queries sharing topics can violate
those assumptions. Results describe the particular dataset and predictions.
Repeated runs and a representative sample are needed to study variability of
a stochastic model. A missed detection does not establish equivalence.
[Paired permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) · [Exact McNemar test](https://online.stat.psu.edu/stat504/Lesson11)

### REST adapter

Install the `rest` extra. Install `statistics` as well to run the paired
comparison shown in the YAML example below.

`RESTAdapter` accepts any JSON endpoint. For `POST`, request mapping keys can
form nested JSON objects. For `GET`, they become query parameters. Paths start
at `input`. `auth_headers` maps HTTP header names to environment variable
names. Each environment variable contains the entire header value, including
`Bearer ` for bearer credentials. Timeout is in seconds and applies per HTTP
operation. `retries` adds zero to three attempts for transient HTTP and
transport errors and defaults to zero. Enable POST retries only when repeating
the request is safe for your service. [HTTPX timeout behavior](https://www.python-httpx.org/advanced/timeouts/)

```python
from llmguard import RESTAdapter

with RESTAdapter(
    "http://localhost:8000/search",
    request_mapping={"request.query": "input"},
    response_mapping={"retrieved_ids": "data.hits.*.id"},
    timeout=3,
) as service:
    evaluator.evaluate(service).report("results/rest.json")
```

The same adapter works in suite YAML. The dataset path is relative to the YAML.

```yaml
task: retrieval
dataset: test.jsonl
metrics: [recall@5, mrr@5, ndcg@5]
model_version: rag-v2
prompt_version: search-v1
quality_gate:
  max_latency_ms: 800
  max_error_rate: 0.02
execution:
  max_concurrency: 4
  requests_per_second: 8
statistics:
  correction: holm
  enforce: false
rest:
  url: https://your-service.example/search
  method: POST
  request_mapping:
    request.query: input
  response_mapping:
    retrieved_ids: data.hits.*.id
  auth_headers:
    Authorization: RAG_AUTHORIZATION
  timeout: 3
  retries: 0
```

```powershell
llmguard suite --config rag-v1.yaml --output results/rest-baseline.json
llmguard suite --config rag-v2.yaml --baseline results/rest-baseline.json --output results/rest-candidate.json
```

### GitHub Actions

Use the repository's Quick Start script or your own evaluation entry point in a workflow.
The SDK example exits with `1` on `FAIL`. The REST suite CLI exits with `0` for
`PASS` or `WARN`, `1` for `FAIL`, and `2` for setup errors. For a deployed REST
service, set `env: {RAG_AUTHORIZATION: "${{ secrets.RAG_AUTHORIZATION }}"}` on
the evaluation step and use the REST suite command as `run`.
[GitHub Actions Python guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)

```yaml
name: AI regression
on: [pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install -e .
      - run: python examples/quickstart.py
```

## BANKING77 evaluation

```powershell
python -m pip install -e ".[banking77]"
# To run the full project test suite:
python -m pip install -e ".[dev]"
python -m pytest -q
```

For live BANKING77 runs, configure `OPENAI_API_KEY`, `OPENAI_CLASSIFIER_MODEL`, and optionally `OPENAI_JUDGE_MODEL`. The judge model defaults to the classifier model.

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

## Generic evaluation suites

The `suite` command accepts task-specific user data in JSON, JSONL or CSV.
It can score precomputed predictions, a chat-completions service, or a REST
service. These workflows use their own JSON and HTML artifacts.

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
python -m pip install -e ".[yaml]"
llmguard suite --config examples/suite/retrieval.yaml --output results/retrieval-baseline.json
llmguard suite --config examples/suite/retrieval-candidate.yaml --output results/retrieval-candidate.json --baseline results/retrieval-baseline.json
```

The candidate intentionally regresses. Both JSON and HTML are written; the
comparison returns exit code 1 on a regression or prediction error, and 2
on invalid configuration. The baseline must be error-free and match the
task, scoring profile, dataset fingerprint, and retrieval depths. Per-metric
tolerances are absolute score drops from 0 to 1 and default to zero.

### Benchmark scoring profiles

Set `profile: squad_v1` on a QA suite to supply every reference answer as a
nonempty `expected` array. LLMGuard applies SQuAD v1.1 normalization and takes
the maximum score across references for each metric. Set `profile: beir` on a
retrieval suite to supply `expected` as an object mapping document IDs to
positive integer relevance grades. Hit, recall, and MRR treat all positive
grades as relevant. nDCG uses the grade as its gain, as in `trec_eval`.
Any retrieval suite can set a single integer `k` or a list such as
`k: [1, 3, 5, 10]`. In CSV, encode arrays and objects as JSON inside quoted cells.
Omitting `profile` uses the default metrics listed above.

### Live model evaluation

Install the `openai` extra for this suite.

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

For offline retrieval, export ranked document IDs from your pipeline or pass
the retriever to the Python SDK. The suite's JSON/HTML and SDK JSON baselines
have separate schemas from the BANKING77 SQLite and baseline formats.
