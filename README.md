# ieee-fraud-features

Engineer Chronon features for credit card fraud detection on the IEEE-CIS dataset. Score is AUC-PR via Lasso (L1 Logistic Regression) on a time-based holdout.

Features are compiled, backfilled, and scored on a shared Chronon instance. Each agent gets an isolated namespace derived from `hive auth whoami`.

## Quickstart

```bash
bash prepare.sh          # Verify Chronon connectivity, create agent namespace
bash eval/eval.sh        # Upload defs → compile → backfill → train → score
# Edit chronon_defs/, re-run eval/eval.sh, repeat
```

## What you modify

- `chronon_defs/group_bys/ieee_fraud/*.py` — Chronon GroupBy definitions (aggregations)
- `chronon_defs/joins/ieee_fraud/fraud_features.py` — Chronon Join definition

## Requirements

- `hive` CLI (authenticated)
- `railway` CLI linked to the Chronon project (or set `CHRONON_CMD` env var)
- Kaggle credentials for data download (data is pre-loaded on the server)
