# Public benchmark methods

## Reproduce the public-dataset benchmarks

Run these commands from the repository root in a Python 3.11 virtual environment.
The official BEIR scorer uses the prebuilt `pytrec_eval` binding from Terrier.
Install the evaluator packages without their retriever dependencies because the
experiment supplies its own BM25 and Dense rankings.

```powershell
python -m pip install -e ".[benchmarks]" -r benchmarks/requirements.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps beir==2.2.0 pytrec-eval-terrier==0.5.11
$run = 'results/public-benchmarks'
python -m benchmarks.download $run
python -m benchmarks.prepare $run
```

**SQuAD v1.1.** The preparation step flattens the original validation JSON in
file order and selects 100 unique questions with `random.Random(42).sample`.
Each suite row sends the passage and question as input and preserves every gold
answer for scoring. Both variants use the same OpenAI model snapshot. The
baseline requests a short answer span and the candidate requests a full
explanatory sentence to exercise regression detection. The official evaluator
receives the same predictions and sampled questions. Its 0–100 scores are
converted to 0–1 before comparison. [SQuAD validation data](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json) · [SQuAD v1.1 evaluation script](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

Run `estimate` before sending requests. It reports estimated input tokens,
maximum output tokens, and USD at the configured rates. That command may
download tokenizer data on its first use. The two `qa` commands send the public
passages and questions to OpenAI using `OPENAI_API_KEY` from the environment.
The JSON reports include returned input, cached input, and output token counts,
request time, and a cost estimate based on those counts and the listed rates.
[Model and rates](https://developers.openai.com/api/docs/models/gpt-5.4-nano)

```powershell
python -m benchmarks.evaluate $run estimate
# Set OPENAI_API_KEY in the environment before generating answers.
python -m benchmarks.evaluate $run qa baseline
python -m benchmarks.evaluate $run qa candidate
python -m benchmarks.evaluate $run score squad
```

Answer generation saves each response and its usage as it completes. Running
the same `qa` command against the same output folder reuses completed responses
and resumes missing ones. `score` and `verify` can replay saved answers without
an API key. Sampling and replay are deterministic. Fresh hosted responses can
vary, and the pinned model's future availability depends on OpenAI.

**BEIR NFCorpus.** The experiment uses all 3,633 documents and the 323 queries
in BEIR's test relevance judgments. BM25Okapi indexes the combined title and
text with lowercase word tokens, `k1=1.5`, and `b=0.75`. Dense retrieval uses the
pinned MiniLM checkpoint with attention-masked mean pooling, 128-token inputs,
L2-normalized embeddings, and exact cosine similarity on CPU. Ties are resolved
by document ID. Both methods produce top-10 rankings that are scored at 1, 3,
5, and 10. BEIR's `top_k_accuracy` is the reference for Hit@K. BEIR rounds all
reference metrics to five decimals, so the verifier allows at most half that
rounding unit plus floating-point slack. [BEIR dataset](https://huggingface.co/datasets/BeIR/nfcorpus) · [Test judgments](https://huggingface.co/datasets/BeIR/nfcorpus-qrels) · [BEIR evaluator](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py) · [Dense model revision](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/commit/e8f8c211226b894fcb81acc59f3b34ba3efd5f42)

```powershell
python -m benchmarks.evaluate $run retrieve bm25
python -m benchmarks.evaluate $run retrieve dense
python -m benchmarks.evaluate $run score nfcorpus
# For an independent ranking replay after the model is cached
python -m benchmarks.evaluate $run retrieve bm25 --repeat
python -m benchmarks.evaluate $run retrieve dense --offline-model --repeat
python -m benchmarks.verify $run
```

Start with `verification.html` or `verification.json` in the run folder. The
`squad` and `nfcorpus` subfolders contain individual JSON and HTML reports,
prepared data, saved predictions, and data hashes. Verification compares to the
official scorers and reports the original generic metrics for the same
predictions. It replays saved predictions in reverse input order, checks for
identical results, and compares fresh retrieval rankings when repeat files are
present. Zero-tolerance gates flag metric drops. The truncated BM25 control
keeps top-1 scores fixed and tests that degradation at deeper cutoffs is
detected. Comparing a run with itself must pass. The `score` stage writes
candidate reports even when their quality gate fails, and the `verify` stage
exits nonzero on a failed verification check. Timing fields report preparation,
API requests or local index and query time, framework scoring, and reference
scoring separately.

### Reproduce efficiency and statistical validation

After the original public benchmark artifacts exist under `$run`, replay
them and measure sequential versus parallel execution on a local simulated
service. The benchmark writes paired results for the original comparisons,
identical controls, and deliberately degraded predictions. Its sensitivity
table reports detection counts over 12 seeded trials for each sample size
and corruption fraction. Fractions describe inputs modified, and measured
effects describe the resulting score changes. The trials reuse the same
source cases, so the detection counts describe these simulations.

```powershell
python -m benchmarks.efficient_evaluation $run --output results/efficient-evaluation/verification.json
```

This replay uses saved predictions and a simulated HTTP transport. It runs
without model credentials and produces a sibling `verification.html`.
