# Local Validation

Validated on 2026-09-13 with the clean project virtual environment on Windows.

| Check | Result |
|---|---|
| `python -m pytest -q` | 88 passed in 7.91 seconds |
| CLI help, evaluate help, compare help | Exit 0 |
| BANKING77 loader and canonical ClassLabel mapping | Covered by loader tests; official CSV revision pinned |
| Sampling | Seeded stratified round-robin; smoke covers all 77 intents; regression is nested for the same seed |
| Offline v1 baseline | PASS, 100 fixtures, accuracy 95.00%, Macro F1 94.29%, judge moderate 5, exit 0 |
| Offline v2 comparison | FAIL, accuracy 88.00% (-7.00 pp), Macro F1 84.42% (-9.87 pp), 10 regressions, 3 improvements, exit 1 |
| Selective Judge, SQLite, JSON/HTML artifacts | Verified in offline demo and integration tests |
| Report rendering | Generated responsive HTML; desktop/mobile smoke check passed previously |
| Provider audit | OpenAI is the sole live provider and no alternate provider artifacts remain |

No real or large paid API call was made. The offline fixtures are synthetic and do not represent model benchmark performance. A live OpenAI run, Docker build, and GitHub-hosted workflow remain unexecuted.
