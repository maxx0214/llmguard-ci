"""Compare two local retrievers on synthetic examples, with no API calls."""

from llmguard import Evaluator, System


cases = [
    {"id": "return", "input": "How do I return an order?", "expected": ["returns"]},
    {"id": "shipping", "input": "When will my order ship?", "expected": ["shipping"]},
    {"id": "tracking", "input": "How do I track a shipment?", "expected": ["tracking"]},
]
rankings = {case["input"]: case["expected"] for case in cases}


def baseline(question):
    return rankings[question]


def candidate(question):
    if question == cases[0]["input"]:
        return ["shipping", "returns"]
    return rankings[question]


if __name__ == "__main__":
    evaluator = Evaluator(
        cases, "retrieval", ["hit@1", "recall@3"],
        quality_gate={"max_drop": {"hit@1": {"warn": 0.10, "fail": 0.50}}},
    )
    result = evaluator.compare(System(baseline, model_version="search-v1"),
                               System(candidate, model_version="search-v2"))
    result.report("results/quickstart.json")
    print(f"{result.status}: {result.metrics}")
    raise SystemExit(1 if result.status == "FAIL" else 0)
