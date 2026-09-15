# Migration Notes

This repository is now BANKING77-first and OpenAI-only. The former provider boundary remains as a typed structured client, with the OpenAI adapter as the sole live implementation and an explicit offline fixture client for deterministic demos.

The official PolyAI CSV files are loaded through Hugging Face `datasets` at a pinned commit. Sampling is seeded stratified round-robin with deterministic class coverage: smoke contains all 77 intents, regression is a same-seed superset, and full uses all 3,080 test rows.

The evaluator records authoritative accuracy and Macro F1, weighted F1, latency, token usage, errors, and selective LLM Judge results. SQLite snapshots are versioned and immutable; comparisons produce explicit regression, improvement, changed-failure, and stable transitions. Reports are escaped HTML artifacts.

Windows cache roots longer than 180 characters are mapped to a stable hashed short directory under the system temp directory. This avoids invalid `datasets` lock filenames while preserving cache identity.

Live calls require only `OPENAI_API_KEY`; model selection uses `OPENAI_CLASSIFIER_MODEL` and optional `OPENAI_JUDGE_MODEL`.
