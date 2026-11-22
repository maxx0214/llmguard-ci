# Contributing to LLMGuard CI

Create a feature branch. Install the project and development extras in a
virtual environment, then run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
python -m twine check --strict dist/*
python scripts/check_release.py dist
```

The test suite uses fixtures and mocked services. REST tests open loopback
sockets. Use public or synthetic fixtures for new tests and omit API keys,
private input data, and generated reports from commits. Keep public benchmark
downloads and paid experiments separate from routine tests.
Install `langchain-core>=0.3.86,<1` to include the optional LangChain integration test.

Record user-facing changes in `CHANGELOG.md`. Metric changes should retain a
reference check and update the reported metric version if their scoring
semantics change. For packaging changes, install the built wheel in a fresh
virtual environment without extras and run the SDK Quick Start.
