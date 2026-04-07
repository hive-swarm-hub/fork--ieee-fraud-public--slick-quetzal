#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# --- Agent identity (immutable, from hive) ---
AGENT_NAME=$(hive auth whoami 2>&1) || {
    echo "ERROR: 'hive auth whoami' failed. Register with hive first."
    exit 1
}
if [ -z "$AGENT_NAME" ] || echo "$AGENT_NAME" | grep -qi "error\|not found\|unauthorized"; then
    echo "ERROR: Invalid hive identity: '$AGENT_NAME'"
    exit 1
fi

NAMESPACE=$(echo "$AGENT_NAME" | tr '-' '_' | tr '[:upper:]' '[:lower:]')
TEAM="ieee_fraud_${NAMESPACE}"

# --- Configuration ---
# Auto-link Railway project if not already linked
if command -v railway >/dev/null 2>&1; then
    if ! railway status >/dev/null 2>&1; then
        echo "Linking Railway to chronon project..."
        railway link -w rLLM -p chronon -e production >/dev/null 2>&1 || {
            echo "WARNING: Failed to auto-link Railway."
        }
    fi
fi
CHRONON_CMD="${CHRONON_CMD:-railway ssh --service main --}"
CHRONON_DIR="${CHRONON_DIR:-/srv/chronon}"
SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-4G}"
BACKFILL_END_DATE="${BACKFILL_END_DATE:-2018-06-03}"

echo "=== IEEE-CIS Fraud Features — Chronon Eval ==="
echo "Agent: $AGENT_NAME | Team: $TEAM"

# --- Step 1: Upload Chronon definitions with agent-scoped team ---
echo ""
echo "[1/6] Uploading Chronon definitions (team: $TEAM)..."

for dir in group_bys joins; do
    for f in chronon_defs/$dir/ieee_fraud/*.py; do
        [ -f "$f" ] || continue
        BASENAME=$(basename "$f")
        REMOTE_PATH="$CHRONON_DIR/$dir/$TEAM/$BASENAME"
        CONTENT=$(cat "$f")
        $CHRONON_CMD "mkdir -p $CHRONON_DIR/$dir/$TEAM && cat > $REMOTE_PATH << 'CHRONON_DEF_EOF'
$CONTENT
CHRONON_DEF_EOF"
        echo "  Uploaded $dir/$TEAM/$BASENAME"
    done
done

# --- Step 2: Compile ---
echo ""
echo "[2/6] Compiling join definition..."
$CHRONON_CMD "cd $CHRONON_DIR && echo n | compile.py --conf=joins/$TEAM/fraud_features.py" 2>&1 | tail -10

# --- Step 3: Backfill ---
echo ""
echo "[3/6] Running Spark backfill (this may take several minutes)..."
$CHRONON_CMD "cd $CHRONON_DIR && spark-submit --class ai.chronon.spark.Driver --master local[*] --driver-memory $SPARK_DRIVER_MEMORY --executor-memory 2G /srv/spark/spark_embedded.jar join --conf-path=production/joins/$TEAM/fraud_features.v1 --end-date=$BACKFILL_END_DATE" 2>&1 | tail -30

# --- Step 4: Export backfilled features as CSV ---
echo ""
echo "[4/6] Exporting backfilled features..."

# The backfill output table follows the pattern: <team>_<join_name>_v1
# It lands in the namespace database (ieee_fraud) or default
EXPORT_PY=$(cat << PYEOF
import pandas as pd, os, glob

# Search for the agent-scoped backfill output
search_bases = ["/opt/spark/data/ieee_fraud.db", "/opt/spark/data"]
target_pattern = "${TEAM}_fraud_features_v1"
output_dir = None

for base in search_bases:
    if not os.path.isdir(base):
        continue
    candidate = os.path.join(base, target_pattern)
    if os.path.isdir(candidate):
        output_dir = candidate
        break
    # Fallback: search for any matching dir
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d)
        if "${TEAM}" in d and "fraud_features" in d and not d.endswith("_bootstrap") and os.path.isdir(full):
            parquets = glob.glob(f"{full}/**/*.parquet", recursive=True)
            if parquets:
                output_dir = full
                break
    if output_dir:
        break

if not output_dir:
    print("ERROR: No backfill output found for team $TEAM")
    print("Searched for:", target_pattern)
    exit(1)

print(f"Reading from: {output_dir}")
df = pd.read_parquet(output_dir)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")
df.to_csv("/tmp/fraud_features_${NAMESPACE}.csv", index=False)
print(f"Exported to /tmp/fraud_features_${NAMESPACE}.csv")
PYEOF
)

$CHRONON_CMD "cat > /tmp/export_features_${NAMESPACE}.py << 'PYEOF'
$EXPORT_PY
PYEOF
python3 /tmp/export_features_${NAMESPACE}.py" 2>&1

# --- Step 5: Run scoring on server ---
echo ""
echo "[5/6] Training Lasso and scoring on server..."

# Patch the scoring script to read from the agent-scoped CSV
SCORE_PY=$(cat eval/train_and_score_chronon.py | sed "s|/tmp/fraud_features_backfill.csv|/tmp/fraud_features_${NAMESPACE}.csv|g")

$CHRONON_CMD "cat > /tmp/train_score_${NAMESPACE}.py << 'PYEOF'
$SCORE_PY
PYEOF
python3 /tmp/train_score_${NAMESPACE}.py" 2>&1 | tee run.log

# --- Step 6: Parse results ---
echo ""
echo "[6/6] Results:"

SCORE=$(grep "^auc_pr:" run.log | awk '{print $2}')
N_FEAT=$(grep "^n_features:" run.log | awk '{print $2}')
N_NONZERO=$(grep "^n_nonzero:" run.log | awk '{print $2}')
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if [ ! -f results.tsv ]; then
    echo -e "timestamp\tauc_pr\tn_features\tn_nonzero" > results.tsv
fi
echo -e "${TIMESTAMP}\t${SCORE}\t${N_FEAT}\t${N_NONZERO}" >> results.tsv
echo "Score appended to results.tsv"
