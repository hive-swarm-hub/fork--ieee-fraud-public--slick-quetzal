#!/bin/bash
set -e

echo "=== IEEE-CIS Fraud Features — Chronon Setup ==="

# --- Get agent identity (must be registered with hive) ---
AGENT_NAME=$(hive auth whoami 2>&1) || {
    echo "ERROR: 'hive auth whoami' failed. Register with hive first."
    echo "  Run: hive auth login"
    exit 1
}
if [ -z "$AGENT_NAME" ] || echo "$AGENT_NAME" | grep -qi "error\|not found\|unauthorized"; then
    echo "ERROR: Invalid hive identity: '$AGENT_NAME'"
    echo "  Run: hive auth login"
    exit 1
fi
echo "Agent: $AGENT_NAME"

# Derive Chronon namespace from agent name (replace hyphens with underscores for Hive compatibility)
NAMESPACE=$(echo "$AGENT_NAME" | tr '-' '_' | tr '[:upper:]' '[:lower:]')
echo "Chronon namespace: ieee_fraud_${NAMESPACE}"

# --- Configuration ---
# Auto-link Railway project if not already linked
if [ -z "$CHRONON_CMD" ] && command -v railway >/dev/null 2>&1; then
    if ! railway status >/dev/null 2>&1; then
        echo "Linking Railway to chronon project..."
        railway link -w rLLM -p chronon -e production >/dev/null 2>&1 || {
            echo "WARNING: Failed to auto-link Railway. Set CHRONON_CMD manually."
        }
    fi
fi
CHRONON_CMD="${CHRONON_CMD:-railway ssh --service main --}"

# Check Chronon connectivity
echo ""
echo "Checking Chronon server connectivity..."
$CHRONON_CMD echo "connected" 2>/dev/null || {
    echo "ERROR: Cannot reach Chronon server."
    echo "Set CHRONON_CMD to your access command, e.g.:"
    echo "  export CHRONON_CMD='railway ssh --service main --'"
    exit 1
}
echo "  Connected."

# Check source data exists
echo "Checking ieee_fraud source data..."
$CHRONON_CMD ls /opt/spark/data/ieee_fraud.db/transactions >/dev/null 2>&1 || {
    echo "ERROR: ieee_fraud.transactions not found. Load source data first."
    exit 1
}
echo "  Source data found."

# Create per-agent Chronon team
echo "Setting up Chronon team: ieee_fraud_${NAMESPACE}..."
TEAM_NAME="ieee_fraud_${NAMESPACE}"

SETUP_PY=$(cat << PYEOF
import json
with open("/srv/chronon/teams.json") as f:
    teams = json.load(f)
if "${TEAM_NAME}" not in teams:
    teams["${TEAM_NAME}"] = {"description": "Agent ${AGENT_NAME}", "namespace": "ieee_fraud"}
    with open("/srv/chronon/teams.json", "w") as f:
        json.dump(teams, f, indent=4)
    print("  Created team ${TEAM_NAME}")
else:
    print("  Team ${TEAM_NAME} already exists")
PYEOF
)

$CHRONON_CMD "cat > /tmp/setup_team.py << 'PYEOF'
$SETUP_PY
PYEOF
python3 /tmp/setup_team.py"

# Create directories
$CHRONON_CMD mkdir -p /srv/chronon/group_bys/${TEAM_NAME} /srv/chronon/joins/${TEAM_NAME}
echo "  Chronon directories ready."

# Check server dependencies
echo "Checking server Python dependencies..."
$CHRONON_CMD pip show scikit-learn pyarrow >/dev/null 2>&1 || {
    echo "  Installing dependencies on server..."
    $CHRONON_CMD pip install -q scikit-learn pyarrow
}
echo "  Dependencies OK."

# Install local dependencies
pip install -q -r requirements.txt 2>/dev/null || true

echo ""
echo "Ready. Run 'bash eval/eval.sh' to establish baseline."
echo "Your Chronon namespace: ieee_fraud_${NAMESPACE}"
