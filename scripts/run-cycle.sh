#!/bin/bash
# Run one cycle of NDIF monitoring
# Intended to be called by cron every 30 minutes

set -e

# Change to script directory
cd "$(dirname "$0")/.."

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Default deploy path (can be overridden via environment variable)
DEPLOY_PATH="${NDIF_DEPLOY_PATH:-}"

# Build deployment args
DEPLOY_ARGS=""
if [ -n "$DEPLOY_PATH" ]; then
    DEPLOY_ARGS="--dashboard --deploy $DEPLOY_PATH"
fi

# Run one model in cycle mode
python run_monitor.py --cycle --no-save $DEPLOY_ARGS

echo "Cycle completed at $(date)"
