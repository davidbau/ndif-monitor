#!/bin/bash
# Run one cycle of NDIF monitoring
# Intended to be called by cron every 30 minutes

set -e

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Find and activate conda environment (requires Python 3.12 for NDIF compatibility)
CONDA_SH=""
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
fi

if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate ndif-monitor-312
elif [ -f "venv/bin/activate" ]; then
    # Fallback to regular venv (may not have Python 3.12)
    source venv/bin/activate
else
    echo "ERROR: Conda not found. NDIF requires Python 3.12."
    echo ""
    echo "Run the setup script to install:"
    echo "  $SCRIPT_DIR/setup.sh"
    exit 1
fi

# Verify Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo "WARNING: Python $PYTHON_VERSION detected, but NDIF server requires 3.12"
    echo "Tests may fail with version mismatch errors."
fi

# Reinstall packages daily to catch package regressions
LAST_INSTALL_FILE="$SCRIPT_DIR/../.last_package_install"
REINSTALL_INTERVAL_HOURS=${REINSTALL_INTERVAL_HOURS:-24}

should_reinstall() {
    if [ ! -f "$LAST_INSTALL_FILE" ]; then
        return 0  # Never installed, do it
    fi
    local last_install=$(cat "$LAST_INSTALL_FILE")
    local now=$(date +%s)
    local age_hours=$(( (now - last_install) / 3600 ))
    [ "$age_hours" -ge "$REINSTALL_INTERVAL_HOURS" ]
}

if should_reinstall; then
    echo "Reinstalling packages (fresh install every ${REINSTALL_INTERVAL_HOURS}h)..."
    pip install --quiet --upgrade --force-reinstall nnsight
    pip install --quiet --upgrade papermill ipykernel torch
    date +%s > "$LAST_INSTALL_FILE"
    echo "Package reinstall complete."
    echo ""
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
