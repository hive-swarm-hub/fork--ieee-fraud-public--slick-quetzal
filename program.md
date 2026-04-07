# IEEE-CIS Fraud Feature Engineering (Chronon)

Design Chronon GroupBy/Join feature definitions that maximize fraud detection AUC-PR using a Lasso (L1 Logistic Regression) model. Features are compiled, backfilled, and scored on a live Chronon instance.

## Setup

1. **Read the in-scope files**:
   - `chronon_defs/group_bys/ieee_fraud/*.py` — Chronon GroupBy definitions. You modify these.
   - `chronon_defs/joins/ieee_fraud/fraud_features.py` — Chronon Join definition. You modify this.
   - `eval/eval.sh` — Uploads defs, compiles, backfills, scores. Do not modify.
   - `eval/train_and_score_chronon.py` — Trains Lasso on backfilled features. Do not modify.
2. **Run prepare**: `bash prepare.sh` to verify Chronon connectivity and data.
3. **Initialize results.tsv**: Create `results.tsv` with just the header row.
4. **Run baseline**: `bash eval/eval.sh` to establish the starting score.

## The benchmark

The IEEE-CIS Fraud Detection dataset has 590,540 transactions (3.5% fraud) over 183 days, loaded into a Chronon Spark warehouse as `ieee_fraud.transactions` and `ieee_fraud.identity`. The eval compiles your Chronon definitions, runs a Spark backfill to materialize point-in-time correct features, then trains a Lasso on the backfilled join table.

The model is deliberately simple (Lasso) — **feature quality is everything**. If your Chronon features lift AUC-PR on Lasso, they are genuinely informative and production-ready.

## Chronon data schema

### ieee_fraud.transactions (590,540 rows, partitioned by ds)
- `TransactionID` — unique ID
- `isFraud` — target (0/1)
- `TransactionDT` — seconds from reference timestamp
- `ts` — epoch milliseconds (derived from TransactionDT)
- `ds` — date partition (yyyy-MM-dd)
- `TransactionAmt` — amount in USD
- `ProductCD` — product code (W, C, H, S, R)
- `card1`–`card6` — card identifiers and metadata
- `addr1`, `addr2` — billing address
- `C1`–`C14`, `D1`–`D15`, `M1`–`M9`, `V1`–`V339` — Vesta features

### ieee_fraud.identity (144,233 rows, partitioned by ds)
- `TransactionID` — join key
- `id_01`–`id_38`, `DeviceType`, `DeviceInfo`
- `ts`, `ds` — inherited from transaction join

## Experimentation

**What you CAN do:**
- Add/modify GroupBy files in `chronon_defs/group_bys/ieee_fraud/`
- Modify the Join in `chronon_defs/joins/ieee_fraud/fraud_features.py`
- Add new GroupBys with different keys (card1, addr1, P_emaildomain, etc.)
- Use any Chronon aggregation: SUM, COUNT, AVERAGE, MIN, MAX, VARIANCE, UNIQUE_COUNT, LAST_K(n), APPROX_PERCENTILE
- Use any window sizes (1, 3, 7, 14, 30, 90 days)
- Add Derivations to the Join for ratios, z-scores, differences between windows
- Create GroupBys over the identity table

**What you CANNOT do:**
- Modify `eval/` or `prepare.sh`
- Aggregate `isFraud` in GroupBys (that would leak the target)
- Use windows that look into the future

**The goal: maximize AUC-PR.** Higher is better. The eval uses a time-based 75/25 split (TransactionDT ≤ 11246620 for train).

## Chronon patterns that work for fraud

**Velocity features** — transaction count per card over short windows (3d, 7d):
```python
Aggregation(input_column="TransactionAmt", operation=Operation.COUNT,
            windows=[Window(length=3, timeUnit=TimeUnit.DAYS)])
```

**Amount deviation** — compare current amount to historical average:
```python
# In GroupBy: average per card over 30d
# In Join derivation: z-score
Derivation(name="amt_zscore_30d",
           expression="(TransactionAmt - ieee_fraud_txn_features_v1_TransactionAmt_average_30d) / ...")
```

**Multi-key GroupBys** — aggregate by different entities (card, address, email):
```python
# Separate GroupBy files for card1, addr1, P_emaildomain keys
```

**Cross-window ratios** — 3d count / 30d count reveals sudden spikes:
```python
Derivation(name="velocity_ratio_3d_30d",
           expression="ieee_fraud_txn_features_v1_TransactionAmt_count_3d / ieee_fraud_txn_features_v1_TransactionAmt_count_30d")
```

## Output format

```
---
auc_pr:           0.4523
n_features:       42
n_nonzero:        28
train_auc_pr:     0.4891
```
