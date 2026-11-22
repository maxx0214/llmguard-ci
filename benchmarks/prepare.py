"""Convert the public benchmark splits into LLMGuard suite inputs."""
import argparse
import csv
import json
import random
from pathlib import Path
from time import perf_counter

from .download import SHA256, sha256


SEED = 42
SQUAD_SIZE = 100
NFCORPUS_COUNTS = (3633, 323)
DEPTHS = [1, 3, 5, 10]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def prepare_squad(root: Path) -> None:
    started = perf_counter()
    raw = root / "raw/dev-v1.1.json"
    if sha256(raw) != SHA256[raw.name]:
        raise ValueError("SQuAD validation checksum mismatch")
    source = read_json(raw)
    items = [(article["title"], paragraph["context"], qa) for article in source["data"]
             for paragraph in article["paragraphs"] for qa in paragraph["qas"]]
    if source["version"] != "1.1" or len(items) != 10570 or len({qa["id"] for _, _, qa in items}) != len(items):
        raise ValueError("Unexpected SQuAD v1.1 validation split")
    selected = random.Random(SEED).sample(items, SQUAD_SIZE)
    rows = [{"id": qa["id"], "input": f"Context:\n{context}\n\nQuestion:\n{qa['question']}",
             "expected": [answer["text"] for answer in qa["answers"]]} for _, context, qa in selected]
    official_sample = {"version": "1.1", "data": [
        {"title": title, "paragraphs": [{"context": context, "qas": [qa]}]}
        for title, context, qa in selected]}
    folder = root / "squad"
    write_json(folder / "dataset.json", rows)
    write_json(folder / "official-sample.json", official_sample)
    write_json(folder / "dataset-meta.json", {
        "split": "validation", "population": len(items), "size": len(rows), "seed": SEED,
        "input_sha256": sha256(raw), "dataset_sha256": sha256(folder / "dataset.json"),
        "preparation_seconds": perf_counter() - started,
    })


def prepare_nfcorpus(root: Path) -> None:
    import pyarrow.parquet as parquet

    started = perf_counter()
    raw = root / "raw"
    input_hashes = {name: sha256(raw / name) for name in ("corpus.parquet", "queries.parquet", "test.tsv")}
    if any(digest != SHA256[name] for name, digest in input_hashes.items()):
        raise ValueError("NFCorpus input checksum mismatch")
    documents = parquet.read_table(raw / "corpus.parquet", columns=["_id", "title", "text"]).to_pylist()
    queries = parquet.read_table(raw / "queries.parquet", columns=["_id", "text"]).to_pylist()
    corpus = {row["_id"]: f"{row['title'] or ''} {row['text']}".strip() for row in documents}
    query_text = {row["_id"]: row["text"] for row in queries}
    qrels = {}
    with (raw / "test.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            query, doc, grade = row["query-id"], row["corpus-id"], int(row["score"])
            judgments = qrels.setdefault(query, {})
            if doc in judgments or doc not in corpus or query not in query_text:
                raise ValueError("Unexpected NFCorpus judgment IDs")
            judgments[doc] = grade
    if (len(corpus), len(qrels)) != NFCORPUS_COUNTS or len(corpus) != len(documents) or len(query_text) != len(queries):
        raise ValueError("Unexpected NFCorpus test split or corpus size")
    rows = [{"id": query, "input": query_text[query],
             "expected": {doc: grade for doc, grade in sorted(qrels[query].items()) if grade > 0}}
            for query in sorted(qrels)]
    if any(not row["expected"] for row in rows):
        raise ValueError("NFCorpus test query has no positive judgments")
    folder = root / "nfcorpus"
    write_json(folder / "dataset.json", rows)
    write_json(folder / "corpus.json", {doc: corpus[doc] for doc in sorted(corpus)})
    write_json(folder / "qrels.json", qrels)
    write_json(folder / "dataset-meta.json", {
        "split": "test", "queries": len(rows), "corpus_documents": len(corpus), "depths": DEPTHS,
        "relevance_grades": sorted({grade for judged in qrels.values() for grade in judged.values()}),
        "input_sha256": input_hashes,
        "dataset_sha256": sha256(folder / "dataset.json"),
        "corpus_sha256": sha256(folder / "corpus.json"),
        "preparation_seconds": perf_counter() - started,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset", choices=("squad", "nfcorpus", "both"), default="both")
    args = parser.parse_args()
    if args.dataset in ("squad", "both"):
        prepare_squad(args.root)
    if args.dataset in ("nfcorpus", "both"):
        prepare_nfcorpus(args.root)
