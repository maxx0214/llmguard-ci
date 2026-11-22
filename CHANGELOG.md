# Changelog

## 0.3.0

First public release.

- Evaluate saved JSON, JSONL, and CSV predictions by case ID through the same
  SDK as live systems. Reports preserve file digests and distinguish archived
  measurements from API calls and charges incurred by the replay.
- Map datasets in memory with nested field selectors, including array wildcards.
  Equivalent mapped and canonical datasets use the same fingerprint.
- Record and summarize per-case metadata. Register per-case custom metrics
  with version labels and choose their direction for Quality Gates.
- Compare both systems, execution details, paired statistics, and matched cases
  in a single HTML report.
- Install the core SDK without third-party runtime dependencies. REST, YAML,
  OpenAI, paired statistics, BANKING77, and benchmarks use optional extras.
- Separate REST response caches and checkpoints by authentication context.
  Existing REST state files require a fresh evaluation. Python-callable state
  keeps its existing format.
- Reject inconsistent saved baseline scores before invoking a candidate.
  Keep built-in HTTP request payloads out of provider debug logs.
- Add an offline Quick Start, a synthetic HTML report, a Python 3.11–3.13 CI
  matrix, and a manual publication guide.
