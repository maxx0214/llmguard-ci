"""BANKING77 via Hugging Face datasets' cache and built-in CSV loader.

Modern datasets versions no longer execute dataset scripts. Use exactly the
official CSVs referenced by PolyAI/banking77, pinned to an immutable commit.
No remote Python is executed and no benchmark examples are vendored.
"""
import random
import os
import hashlib
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Literal

from llmguard.config import EvaluationConfig, fingerprint
from llmguard.data.intents import INTENTS
from llmguard.models import Banking77Case, DatasetInfo, Tier

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data"


def cache_path(path: Path | None) -> str | None:
    """Return a datasets-safe cache path, shortening long Windows roots."""
    if path is None:
        return None
    value = str(path.resolve())
    # datasets embeds the cache root in lock filenames, so keep the root well
    # below MAX_PATH even when the caller's workspace is deeply nested.
    if os.name == "nt" and len(value) > 80:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
        value = str(Path(tempfile.gettempdir()) / f"llmguard-hf-{digest}")
        Path(value).mkdir(parents=True, exist_ok=True)
    return value


def select_indices(labels: list[str], count: int, seed: int) -> list[int]:
    """Seeded class-balanced selection; smaller tiers are subsets of larger tiers."""
    if not labels or count < 1:
        raise ValueError("Nonempty dataset and positive sample count required")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    ordered = []
    while len(ordered) < min(count, len(labels)):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
                if len(ordered) == min(count, len(labels)):
                    break
    return sorted(ordered)


def load_banking77(tier: Tier = "smoke", seed: int = 42,
                   split: Literal["train", "test"] = "test",
                   config: EvaluationConfig | None = None,
                   cache_dir: Path | None = None) -> tuple[list[Banking77Case], DatasetInfo]:
    from datasets import ClassLabel, Dataset, DownloadConfig, DownloadManager, Features, Value, load_dataset
    import csv

    if tier not in ("smoke", "regression", "full") or split not in ("train", "test"):
        raise ValueError("Invalid evaluation tier or official split")
    config = config or EvaluationConfig()
    features = Features({"text": Value("string"), "category": ClassLabel(names=list(INTENTS))})
    download = DownloadManager(download_config=DownloadConfig(
        cache_dir=cache_path(cache_dir / "downloads") if cache_dir else None,
        local_files_only=os.getenv("HF_DATASETS_OFFLINE", "0").lower() in ("1", "true", "yes"),
        use_etag=False))
    paths = download.download({s: f"{SOURCE}/{s}.csv" for s in ("train", "test")})
    local_paths = {key: value for key, value in paths.items()
                   if isinstance(value, str) and Path(value).is_file()}
    if len(local_paths) == len(paths):
        # Avoid the packaged builder's Windows drive-letter parsing for local
        # files returned by DownloadManager; the same schema is preserved.
        rows = []
        with Path(local_paths[split]).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        dataset = Dataset.from_dict({"text": [row["text"] for row in rows],
                                     "category": [INTENTS.index(row["category"]) for row in rows]})
        dataset = dataset.cast(features)
    else:
        dataset = load_dataset("csv", data_files=paths,
                               features=features, split=split, cache_dir=cache_path(cache_dir))
    mapping = dataset.features["category"]
    if tuple(mapping.names) != INTENTS:
        raise ValueError("BANKING77 ClassLabel mapping differs from canonical taxonomy")
    labels = [mapping.int2str(int(value)) for value in dataset["category"]]
    size = len(dataset) if tier == "full" else getattr(config, f"{tier}_size")
    indices = list(range(len(dataset))) if tier == "full" else select_indices(labels, size, seed)
    cases = [Banking77Case(id=f"BANKING77-{split.upper()}-{i + 1:04d}",
                          text=dataset[i]["text"], expected_intent=labels[i]) for i in indices]
    info = DatasetInfo(split=split, tier=tier, sample_count=len(cases), seed=seed,
                       revision=REVISION, fingerprint=fingerprint([c.model_dump() for c in cases]))
    return cases, info
