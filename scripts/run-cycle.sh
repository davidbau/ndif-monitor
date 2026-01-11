#!/bin/bash
# Run one cycle of NDIF monitoring
# Intended to be called by cron every 30 minutes

set -e

# Change to script directory
cd "$(dirname "$0")/.."

# Activate conda environment (requires Python 3.12 for NDIF compatibility)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ndif-monitor-312
elif [ -f "venv/bin/activate" ]; then
    # Fallback to regular venv
    source venv/bin/activate
fi

# Default deploy path (can be overridden via environment variable)
DEPLOY_PATH="${NDIF_DEPLOY_PATH:-/share/projects/ndif-monitor/www}"

# Build deployment args
DEPLOY_ARGS=""
if [ -n "$DEPLOY_PATH" ]; then
    DEPLOY_ARGS="--dashboard --deploy $DEPLOY_PATH"
fi

# Run one model in cycle mode
python run_monitor.py --cycle --no-save $DEPLOY_ARGS

echo "Cycle completed at $(date)"
