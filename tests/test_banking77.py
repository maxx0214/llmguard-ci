import pytest
from datasets import ClassLabel, Dataset, Features, Value

from llmguard.config import EvaluationConfig
from llmguard.data.banking77 import REVISION, load_banking77, select_indices
from llmguard.data.intents import INTENTS
from llmguard.models import IntentClassificationResult


@pytest.fixture(autouse=True)
def mock_download(monkeypatch):
    monkeypatch.setattr("datasets.DownloadManager.download", lambda self, paths: paths)


def fake_split(n=770):
    return Dataset.from_dict({"text": [f"Query {i}" for i in range(n)], "category": [i % 77 for i in range(n)]},
        features=Features({"text": Value("string"), "category": ClassLabel(names=list(INTENTS))}))


def test_label_source_and_strict_parsing():
    assert len(INTENTS) == len(set(INTENTS)) == 77
    assert INTENTS[11] == "card_arrival"
    assert INTENTS[51] == "Refund_not_showing_up"
    assert INTENTS[53] == "reverted_card_payment?"
    assert IntentClassificationResult(intent=INTENTS[51]).intent == INTENTS[51]
    with pytest.raises(ValueError):
        IntentClassificationResult(intent="refund_not_showing_up")


def test_official_loader_mapping_and_ids(monkeypatch, tmp_path):
    calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return fake_split()
    monkeypatch.setattr("datasets.load_dataset", fake)
    cases, info = load_banking77(cache_dir=tmp_path)
    assert len(cases) == 100
    assert len({c.expected_intent for c in cases}) == 77
    assert info.revision == REVISION
    assert cases[0].id.startswith("BANKING77-TEST-")
    index = int(cases[0].id.rsplit("-",1)[1])-1
    assert cases[0].expected_intent == INTENTS[index % 77]
    assert calls[0][0] == ("csv",)
    assert set(calls[0][1]["data_files"]) == {"train", "test"}
    assert REVISION in calls[0][1]["data_files"]["test"]


def test_deterministic_nested_tiers(monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: fake_split())
    small, a = load_banking77()
    repeat, b = load_banking77()
    large, _ = load_banking77("regression")
    full, _ = load_banking77("full")
    changed, _ = load_banking77(seed=7)
    assert small == repeat and a == b
    assert len(large) == 500 and len(full) == 770
    assert {c.id for c in small} <= {c.id for c in large}
    assert {c.id for c in large} <= {c.id for c in full}
    assert small != changed


def test_train_split_and_size_override(monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: fake_split())
    cases, _ = load_banking77(split="train", config=EvaluationConfig(smoke_size=20))
    assert len(cases) == 20 and all(c.id.startswith("BANKING77-TRAIN-") for c in cases)


def test_invalid_label_mapping(monkeypatch):
    wrong = Dataset.from_dict({"text":["x"],"category":[0]},
        features=Features({"text":Value("string"),"category":ClassLabel(names=["wrong"])}))
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: wrong)
    with pytest.raises(ValueError, match="mapping"):
        load_banking77()


def test_selection_validation():
    with pytest.raises(ValueError):
        select_indices([], 100, 42)


def test_builtin_csv_pipeline_without_network(monkeypatch, tmp_path):
    paths = {}
    for split in ("train", "test"):
        path = tmp_path / f"{split}.csv"
        path.write_text("text,category\nWhere is my card?,card_arrival\nHow old must I be?,age_limit\n", encoding="utf-8")
        paths[split] = str(path)
    monkeypatch.setattr("datasets.DownloadManager.download", lambda self, urls: paths)
    cases, info = load_banking77("full", cache_dir=tmp_path / "cache")
    assert [c.expected_intent for c in cases] == ["card_arrival", "age_limit"]
    assert cases[0].id == "BANKING77-TEST-0001"
    assert info.sample_count == 2
