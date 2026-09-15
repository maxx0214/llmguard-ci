# Reviewed BANKING77 Baselines

Generate a real baseline per evaluation tier, review the report, and commit the
snapshot to the target branch:

```sh
python -m llmguard.cli evaluate --prompt prompts/banking77_v1.yaml --tier smoke --set-baseline --export-baseline baselines/smoke.json
```

Use regression.json or full.json for their respective tiers. Seed, official split,
sample fingerprint and metric contract must match. Provider/model changes are
allowed. A candidate PR cannot replace its own baseline: CI reads this directory
from the target commit. Rebaseline separately when sample selection changes.

No model results are fabricated or supplied as a live baseline. Synthetic offline
snapshots and the old customer-support baseline format are incompatible with live
BANKING77 runs. The --set-baseline flag designates a run; a human still needs to
review its quality. Operationally failed runs cannot become baselines.
