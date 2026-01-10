# NDIF Monitor

End-to-end testing framework for NDIF (National Deep Inference Fabric) and nnsight.

## Quick Start

```bash
# Copy and edit credentials
cp .env.local.example .env.local
# Edit .env.local with your NDIF_API and HF_TOKEN

# Run the monitor (all baseline models + extra hot models)
python run_monitor.py

# Round-robin mode: test one model per run
python run_monitor.py --cycle

# View tracked model statuses
python run_monitor.py --show-status

# Generate dashboard (can view in browser)
python run_monitor.py --cycle --dashboard
open results/dashboard.html

# Just check NDIF status (no tests)
python run_monitor.py --status-only
```

## Requirements

- Python 3.10+
- NDIF API key from [nnsight.net](https://nnsight.net)
- HuggingFace token for gated models (optional)

## Lab Machine Deployment

For running on a dedicated machine every 30 minutes:

### Setup

```bash
# Clone the repo
git clone https://github.com/your-org/ndif-monitor.git
cd ndif-monitor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up credentials
cp .env.local.example .env.local
nano .env.local  # Add your NDIF_API and HF_TOKEN
```

### Install Cron Job

```bash
# Install with dashboard deployment
./scripts/install-cron.sh --deploy /share/projects/ndif-monitor/www

# Or install without deployment (just run tests)
./scripts/install-cron.sh

# Check status
./scripts/install-cron.sh --status

# Uninstall later
./scripts/install-cron.sh --uninstall
```

Options:
- `--deploy PATH` - Deploy dashboard to PATH after each run
- `--interval MIN` - Run every MIN minutes (default: 30)
- `--uninstall` - Remove the cron entry
- `--status` - Show current installation status

### Timing

- **Per cycle**: ~3 minutes (1 model × 3 scenarios)
- **Full coverage**: ~4.5 hours across 9 cycles (all baseline models at 30 min intervals)
- Each run creates a fresh venv for isolation

## Output

### Per-Model Status Files

Each model gets its own JSON file in `results/`:

```
results/
├── openai-community--gpt2.json
├── EleutherAI--gpt-j-6b.json
├── meta-llama--Llama-2-7b-hf.json
├── allenai--Olmo-3-1025-7B.json
└── .cycle_state.json
```

Example model status:
```json
{
  "model": "openai-community/gpt2",
  "last_updated": "2026-01-10T17:00:00Z",
  "overall_status": "OK",
  "last_all_ok": "2026-01-10T17:00:00Z",
  "scenarios": {
    "basic_trace": {
      "status": "OK",
      "duration_ms": 21000,
      "last_checked": "2026-01-10T17:00:00Z",
      "last_success": "2026-01-10T17:00:00Z"
    },
    "generation": { ... },
    "hidden_states": { ... }
  }
}
```

## Baseline Models

These dedicated hot models are always tested (defined in `src/models.py`):

**Small/Fast:**
- `openai-community/gpt2` - GPT-2 (fastest canary)
- `EleutherAI/gpt-j-6b` - GPT-J 6B

**7-8B Models:**
- `meta-llama/Llama-2-7b-hf` - Llama 2 7B
- `meta-llama/Llama-3.1-8B` - Llama 3.1 8B
- `allenai/Olmo-3-1025-7B` - OLMo 7B

**Large Models (70B):**
- `meta-llama/Llama-3.1-70B` - Llama 3.1 70B base
- `meta-llama/Llama-3.1-70B-Instruct` - Llama 3.1 70B instruct
- `meta-llama/Llama-3.3-70B-Instruct` - Llama 3.3 70B instruct

**Very Large (405B):**
- `meta-llama/Llama-3.1-405B-Instruct` - Llama 3.1 405B instruct

Additional hot models from the NDIF status API are also tested.

## Test Scenarios

- **basic_trace**: Tests `model.trace()` with hidden state extraction
- **generation**: Tests text generation via `model.generate()`
- **hidden_states**: Extracts hidden states from multiple layers

## Status Levels

- **OK**: Test passed normally
- **SLOW**: Passed but exceeded time threshold
- **DEGRADED**: Partial failure
- **FAILED**: Test failed with error
- **UNAVAILABLE**: Model not available

## Configuration

### Local Development

Create `.env.local` with:
```
NDIF_API="your_ndif_api_key"
HF_TOKEN="your_hf_token"
```

### GitHub Actions

```yaml
# .github/workflows/monitor.yml
name: NDIF Monitor
on:
  schedule:
    - cron: '*/30 * * * *'  # Every 30 minutes
  workflow_dispatch:

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Run monitor cycle
        env:
          NDIF_API: ${{ secrets.NDIF_API }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          pip install -r requirements.txt
          python run_monitor.py --cycle

      - name: Commit results
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add results/
          git diff --staged --quiet || git commit -m "Update monitor results"
          git push
```

## Dashboard

The monitor generates a web dashboard showing:
- **Calendar heatmap**: Status history over time (like GitHub contributions)
- **Current status**: Per-model status with scenario breakdown
- **Recent failures**: List with Colab links to reproduce

### Generate Dashboard

```bash
# Generate dashboard after tests
python run_monitor.py --cycle --dashboard

# Regenerate from existing history (no tests)
python run_monitor.py --dashboard-only

# View in browser
open results/dashboard.html
```

### Deploy to Web Server

For hosting at `baulab.us/p/ndif-monitor`:

```bash
# One-time deploy
python run_monitor.py --dashboard-only --deploy /share/projects/ndif-monitor/www

# Or with tests
python run_monitor.py --cycle --deploy /share/projects/ndif-monitor/www
```

### Automated Deployment with Cron

Use the install script with `--deploy`:

```bash
./scripts/install-cron.sh --deploy /share/projects/ndif-monitor/www
```

This sets up a cron job that runs every 30 minutes, tests one model, and deploys the updated dashboard.

### Dashboard Files

```
results/
├── dashboard.html          # Main dashboard page (static HTML)
├── data/
│   └── status.json         # Dashboard data (auto-generated)
├── history.jsonl           # Historical data (append-only, ~50MB/year)
└── *.json                  # Per-model status files
```

When deployed:
```
www/
├── index.html              # Dashboard page
└── data/
    ├── status.json         # Dashboard data
    └── models/
        └── *.json          # Per-model status files
```

## Reproducing Failures

When a test fails, the dashboard provides Colab links to reproduce the issue:

1. Click "Open in Colab" on a failure entry
2. The notebook opens with the model pre-configured
3. Run all cells to see the error
4. Debug interactively in Colab

## Storage Estimates

| Data | Size (30-min intervals, 9 models, 3 scenarios) |
|------|-----------------------------------------------|
| Per day | ~1,300 history entries, ~130 KB |
| Per month | ~40K entries, ~4 MB |
| Per year | ~470K entries, ~50 MB |
| Model status files | ~2-5 KB each |
| Dashboard HTML | ~20 KB |
