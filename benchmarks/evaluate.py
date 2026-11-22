"""Generate predictions and score the public datasets with LLMGuard Suite."""
import argparse
import hashlib
import re
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from llmguard.suite import CompatibleProvider, compare_suite, evaluate_suite, load_rows, write_suite_report

from .prepare import DEPTHS, SEED, read_json, write_json


MODEL = "gpt-5.4-nano-2026-03-17"
PRICING = {"input": 0.20, "cached_input": 0.02, "output": 1.25}
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.4-nano"
QA_PROMPTS = {
    "baseline": ("Answer using the supplied context. Return only the shortest exact answer span "
                 "from the context, with no explanation, quotes, or Markdown.", 64),
    "candidate": ("Answer using the supplied context. Write a full explanatory sentence that restates "
                  "the question and provides the answer. Do not return an isolated phrase.", 96),
}
DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DENSE_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
CPU_THREADS = 4
BATCH_SIZE = 32
DENSE_MAX_LENGTH = 128


def estimate_qa(root: Path) -> dict:
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    rows = load_rows(root / "squad/dataset.json")
    user_tokens = sum(len(encoding.encode_ordinary(row["input"])) for row in rows)
    estimates = {}
    for label, (prompt, cap) in QA_PROMPTS.items():
        input_tokens = user_tokens + len(rows) * (len(encoding.encode_ordinary(prompt)) + 12)
        output_tokens = len(rows) * cap
        estimates[label] = {
            "requests": len(rows), "estimated_input_tokens": input_tokens, "capped_output_tokens": output_tokens,
            "estimated_cost_usd": (input_tokens * PRICING["input"] + output_tokens * PRICING["output"]) / 1e6,
        }
    estimate = {
        "model": MODEL, "usd_per_million_tokens": PRICING, "pricing_source": PRICE_SOURCE,
        "variants": estimates, "estimated_cost_usd": sum(run["estimated_cost_usd"] for run in estimates.values()),
    }
    write_json(root / "squad/estimate.json", estimate)
    return estimate


def generate_qa(root: Path, label: str) -> None:
    from openai import APIError

    folder = root / "squad"
    rows = load_rows(folder / "dataset.json")
    # Validate reference data before making billable requests.
    preflight = evaluate_suite("qa", rows, {}, profile="squad_v1")
    prompt, cap = QA_PROMPTS[label]
    config = {"name": "openai", "model": MODEL, "api_key_env": "OPENAI_API_KEY",
              "base_url": "https://api.openai.com/v1", "system_prompt": prompt,
              "reasoning_effort": "none", "max_completion_tokens": cap}
    source = {"model": MODEL, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "max_completion_tokens": cap, "reasoning_effort": "none",
              "dataset_fingerprint": preflight["dataset_fingerprint"]}
    path = folder / f"{label}-predictions.json"
    state = read_json(path) if path.exists() else {"source": source, "records": {}}
    if state["source"] != source:
        raise ValueError(f"Saved predictions at {path} use a different evaluation configuration")
    missing = [row for row in rows if row["id"] not in state["records"]]
    if not missing:
        print(f"{label}: replaying {len(rows)} cached responses", flush=True)
        return
    adapter = CompatibleProvider(config, max_retries=0)
    try:
        for index, row in enumerate(missing, 1):
            started = perf_counter()
            try:
                answer = adapter.predict(row["input"])
            except APIError as exc:
                raise RuntimeError(f"OpenAI request failed: {type(exc).__name__} "
                                   f"(HTTP {getattr(exc, 'status_code', 'n/a')})") from None
            except ValueError:
                answer = None
            elapsed = perf_counter() - started
            usage = adapter.last_usage
            cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            state["records"][row["id"]] = {
                "prediction": answer, "seconds": elapsed, "usage": None if usage is None else {
                    "input_tokens": usage.prompt_tokens, "cached_input_tokens": cached,
                    "output_tokens": usage.completion_tokens,
                },
            }
            write_json(path, state)
            if index == 1 or index % 10 == 0 or index == len(missing):
                print(f"{label}: {len(state['records'])}/{len(rows)} responses saved", flush=True)
    finally:
        adapter.close()


def dense_embeddings(model, tokenizer, texts: list[str], progress: str | None = None):
    import torch
    from tqdm.auto import tqdm

    embeddings = []
    batches = range(0, len(texts), BATCH_SIZE)
    with torch.inference_mode():
        for offset in tqdm(batches, desc=progress, disable=progress is None):
            inputs = tokenizer(texts[offset:offset + BATCH_SIZE], padding=True, truncation=True,
                               max_length=DENSE_MAX_LENGTH, return_tensors="pt", return_token_type_ids=False)
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            means = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            embeddings.append(torch.nn.functional.normalize(means, dim=1))
    return torch.cat(embeddings).numpy()


def generate_retrieval(root: Path, method: str, *, offline_model: bool = False, repeat: bool = False) -> None:
    import numpy as np

    folder = root / "nfcorpus"
    rows = load_rows(folder / "dataset.json")
    corpus = read_json(folder / "corpus.json")
    doc_ids = sorted(corpus)
    documents = [corpus[doc_id] for doc_id in doc_ids]
    queries = [row["input"] for row in rows]
    started = perf_counter()
    if method == "bm25":
        from rank_bm25 import BM25Okapi

        tokenizer = re.compile(r"\w+")
        index = BM25Okapi([tokenizer.findall(text.lower()) for text in documents], k1=1.5, b=0.75)
        index_seconds = perf_counter() - started
        scores = [index.get_scores(tokenizer.findall(query.lower())) for query in queries]
        source = {"algorithm": "BM25Okapi", "rank_bm25": version("rank-bm25"), "k1": 1.5, "b": 0.75}
    else:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, PreTrainedTokenizerFast

        torch.manual_seed(SEED)
        torch.set_num_threads(CPU_THREADS)
        snapshot = snapshot_download(DENSE_MODEL, revision=DENSE_REVISION, local_files_only=offline_model,
                                     allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                                     "tokenizer_config.json", "special_tokens_map.json"])
        tokenizer = PreTrainedTokenizerFast.from_pretrained(snapshot, local_files_only=True)
        model = AutoModel.from_pretrained(snapshot, use_safetensors=True, local_files_only=True).eval()
        doc_embeddings = dense_embeddings(model, tokenizer, documents, "NFCorpus documents")
        index_seconds = perf_counter() - started
        query_embeddings = dense_embeddings(model, tokenizer, queries)
        scores = query_embeddings @ doc_embeddings.T
        source = {"algorithm": "exact cosine similarity", "model": DENSE_MODEL, "revision": DENSE_REVISION,
                  "transformers": version("transformers"), "torch": version("torch"),
                  "device": "cpu", "cpu_threads": CPU_THREADS, "batch_size": BATCH_SIZE,
                  "max_sequence_length": DENSE_MAX_LENGTH, "pooling": "attention-masked mean", "seed": SEED}
    records = {
        row["id"]: [doc_ids[index] for index in np.argsort(-np.asarray(values), kind="stable")[:max(DEPTHS)]]
        for row, values in zip(rows, scores, strict=True)
    }
    elapsed = perf_counter() - started
    artifact = f"{method}-repeat" if repeat else method
    write_json(folder / f"{artifact}-predictions.json", {
        "source": source, "predictions": records, "timing": {"index_seconds": index_seconds,
                                                                "query_seconds": elapsed - index_seconds},
    })
    print(f"{method}: {len(records)} queries over {len(corpus)} documents in {elapsed:.2f}s", flush=True)


def qa_usage(records: dict) -> dict:
    observed = [row["usage"] for row in records.values() if row["usage"] is not None]
    amounts = {key: sum(usage[key] for usage in observed)
               for key in ("input_tokens", "cached_input_tokens", "output_tokens")}
    uncached = amounts["input_tokens"] - amounts["cached_input_tokens"]
    cost = (uncached * PRICING["input"] + amounts["cached_input_tokens"] * PRICING["cached_input"]
            + amounts["output_tokens"] * PRICING["output"]) / 1e6
    return {"requests": len(records), "responses_with_usage": len(observed), **amounts,
            "estimated_cost_usd": cost if observed else None, "usd_per_million_tokens": PRICING,
            "pricing_source": PRICE_SOURCE}


def score(root: Path, dataset: str) -> None:
    folder = root / dataset
    rows = load_rows(folder / "dataset.json")
    qa = dataset == "squad"
    task = "qa" if qa else "retrieval"
    options = {"profile": "squad_v1"} if qa else {"profile": "beir", "k": DEPTHS}
    labels = ("baseline", "candidate") if qa else ("bm25", "dense")
    baseline = None
    for label in labels:
        state = read_json(folder / f"{label}-predictions.json")
        if qa:
            records = state["records"]
            predictions = {case_id: row["prediction"] for case_id, row in records.items()}
            timing = {"api_request_seconds": sum(row["seconds"] for row in records.values())}
            usage = qa_usage(records)
        else:
            predictions = state["predictions"]
            timing = state["timing"]
            usage = {"requests": 0, "estimated_cost_usd": 0.0}
        if {row["id"] for row in rows} != set(predictions):
            raise ValueError(f"{label} prediction IDs do not match the full evaluation split")
        started = perf_counter()
        run = evaluate_suite(task, rows, predictions, **options)
        if qa and run["dataset_fingerprint"] != state["source"]["dataset_fingerprint"]:
            raise ValueError(f"{label} predictions were generated for different QA inputs")
        if baseline is None:
            baseline = run
        else:
            compare_suite(baseline, run, {})
        run.update(source=state["source"], timing={**timing, "scoring_seconds": perf_counter() - started},
                   api_usage=usage)
        write_suite_report(run, folder / f"{label}.json")
        print(f"{dataset} {label}: {run['status']} ({run['error_count']} prediction errors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("estimate")
    qa_parser = commands.add_parser("qa")
    qa_parser.add_argument("variant", choices=tuple(QA_PROMPTS))
    ir_parser = commands.add_parser("retrieve")
    ir_parser.add_argument("method", choices=("bm25", "dense"))
    ir_parser.add_argument("--offline-model", action="store_true")
    ir_parser.add_argument("--repeat", action="store_true", help="Save an independent retrieval replay")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("dataset", choices=("squad", "nfcorpus"))
    args = parser.parse_args()
    if args.command == "estimate":
        result = estimate_qa(args.root)
        print(f"{result['model']}: estimated USD {result['estimated_cost_usd']:.4f} for both variants")
    elif args.command == "qa":
        generate_qa(args.root, args.variant)
    elif args.command == "retrieve":
        generate_retrieval(args.root, args.method, offline_model=args.offline_model, repeat=args.repeat)
    else:
        score(args.root, args.dataset)
