"""Fetch public inputs for the SQuAD and NFCorpus experiments."""
import argparse
import hashlib
import shutil
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen


NFCORPUS_REVISION = "b5026a0e96e8a7ac4f95f482a596389289d46269"
QRELS_REVISION = "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65"
NFCORPUS_URL = f"https://huggingface.co/datasets/BeIR/nfcorpus/resolve/{NFCORPUS_REVISION}"
SOURCES = {
    "dev-v1.1.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    "evaluate-v1.1.py": "https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py",
    "corpus.parquet": f"{NFCORPUS_URL}/corpus/corpus-00000-of-00001.parquet",
    "queries.parquet": f"{NFCORPUS_URL}/queries/queries-00000-of-00001.parquet",
    "test.tsv": f"https://huggingface.co/datasets/BeIR/nfcorpus-qrels/resolve/{QRELS_REVISION}/test.tsv",
}
SHA256 = {
    "dev-v1.1.json": "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
    "evaluate-v1.1.py": "f5a673dbbd173e29e9ea38f1b2091d883583b77b3a4c17144b223fb0f2f9bd09",
    "corpus.parquet": "f2a1c0b570a5efdf23cfa36f5e573d062cc113c6e7251e22c69b323a33ea895e",
    "queries.parquet": "edd2b878b8130fb3b5e3c4b9428ddc1e67fbd6eedf8ae099fca3b87838ba6b29",
    "test.tsv": "f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(root: Path) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = raw / name
        if not path.exists():
            temp = path.with_suffix(path.suffix + ".part")
            with urlopen(url, timeout=90) as response, temp.open("wb") as output:
                shutil.copyfileobj(response, output)
            if sha256(temp) != SHA256[name]:
                raise ValueError(f"Checksum mismatch for {name}")
            temp.replace(path)
        digest = sha256(path)
        if digest != SHA256[name]:
            raise ValueError(f"Checksum mismatch for {name}")
        print(f"{name} SHA-256 {digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    started = perf_counter()
    download(parser.parse_args().root)
    print(f"Download: {perf_counter() - started:.2f}s")
