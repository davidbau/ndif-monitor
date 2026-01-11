#!/bin/bash
# NDIF Monitor Setup Script
#
# This script sets up everything needed to run NDIF Monitor:
# 1. Installs miniconda if not present
# 2. Creates Python 3.12 environment (required by NDIF server)
# 3. Installs required packages
# 4. Optionally sets up cron job
#
# Usage:
#   ./scripts/setup.sh              # Interactive setup
#   ./scripts/setup.sh --cron       # Also install cron job
#   ./scripts/setup.sh --help       # Show help

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV_NAME="ndif-monitor-312"
DEPLOY_PATH="/share/projects/ndif-monitor/www"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
INSTALL_CRON=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --cron)
            INSTALL_CRON=true
            shift
            ;;
        --help|-h)
            echo "NDIF Monitor Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cron    Also install cron job for automatic monitoring"
            echo "  --help    Show this help"
            echo ""
            echo "This script will:"
            echo "  1. Install miniconda (if not present)"
            echo "  2. Create Python 3.12 conda environment"
            echo "  3. Install nnsight, torch, papermill, etc."
            echo "  4. Optionally set up cron job"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "NDIF Monitor Setup"
echo "========================================"
echo ""

# Step 1: Check for conda
log_info "Checking for conda..."

CONDA_PATH=""
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/miniconda3"
    log_info "Found miniconda at $CONDA_PATH"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/anaconda3"
    log_info "Found anaconda at $CONDA_PATH"
elif command -v conda &> /dev/null; then
    CONDA_PATH="$(conda info --base)"
    log_info "Found conda at $CONDA_PATH"
fi

if [ -z "$CONDA_PATH" ]; then
    log_warn "Conda not found!"
    echo ""
    echo "NDIF requires Python 3.12, which is best installed via conda."
    echo ""
    read -p "Would you like to install miniconda? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_error "Conda is required. Please install it manually:"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3"
        exit 1
    fi

    log_info "Downloading miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

    log_info "Installing miniconda to ~/miniconda3..."
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh

    CONDA_PATH="$HOME/miniconda3"

    # Initialize conda for bash
    "$CONDA_PATH/bin/conda" init bash

    log_info "Miniconda installed successfully!"
    echo ""
    log_warn "You may need to restart your shell or run:"
    echo "  source ~/.bashrc"
    echo ""
fi

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Accept TOS if needed (for newer conda versions)
if conda tos status 2>/dev/null | grep -q "not accepted"; then
    log_info "Accepting conda terms of service..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
fi

# Step 2: Create/check conda environment
log_info "Checking for $CONDA_ENV_NAME environment..."

if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    log_info "Environment $CONDA_ENV_NAME already exists"
else
    log_info "Creating $CONDA_ENV_NAME environment with Python 3.12..."
    conda create -y -n "$CONDA_ENV_NAME" python=3.12
    log_info "Environment created!"
fi

# Step 3: Install packages
log_info "Activating $CONDA_ENV_NAME..."
conda activate "$CONDA_ENV_NAME"

log_info "Installing required packages..."
pip install --quiet nnsight torch papermill ipykernel requests pyyaml

# Verify installation
NNSIGHT_VERSION=$(python -c "import nnsight; print(nnsight.__version__)" 2>/dev/null || echo "failed")
PYTHON_VERSION=$(python --version 2>&1)

if [ "$NNSIGHT_VERSION" = "failed" ]; then
    log_error "Failed to install nnsight!"
    exit 1
fi

echo ""
log_info "Installation complete!"
echo "  Python: $PYTHON_VERSION"
echo "  nnsight: $NNSIGHT_VERSION"
echo ""

# Step 4: Check for credentials
log_info "Checking for credentials..."
if [ -f "$PROJECT_DIR/.env.local" ]; then
    log_info "Found .env.local"
else
    log_warn ".env.local not found!"
    echo ""
    echo "Create $PROJECT_DIR/.env.local with your API keys:"
    echo '  NDIF_API="your-ndif-api-key"'
    echo '  HF_TOKEN="your-huggingface-token"'
    echo ""
    echo "Get your NDIF API key at: https://nnsight.net"
    echo ""
fi

# Step 5: Optionally install cron
if $INSTALL_CRON; then
    echo ""
    log_info "Setting up cron job..."
    "$SCRIPT_DIR/install-cron.sh" --interval 30
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To run a test manually:"
echo "  cd $PROJECT_DIR"
echo "  source $CONDA_PATH/etc/profile.d/conda.sh"
echo "  conda activate $CONDA_ENV_NAME"
echo "  python run_monitor.py --cycle --deploy $DEPLOY_PATH"
echo ""
echo "To install the cron job:"
echo "  $SCRIPT_DIR/install-cron.sh"
echo ""
echo "Dashboard will be at: https://baulab.us/p/ndif-monitor/"
echo ""
