"""Dashboard generator for NDIF Monitor.

Generates a static HTML page with:
- Calendar heatmap showing status over time
- Per-model status grid
- Colab links for reproducing failures
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from .history import HistoryStore, get_hostname, get_username
from .results import ModelStatus, model_to_filename


# GitHub repo info for Colab links
GITHUB_REPO = "davidbau/ndif-monitor"  # Update with actual repo
GITHUB_BRANCH = "main"


def generate_colab_url(notebook: str, model: str) -> str:
    """Generate a Google Colab URL for a notebook with model preset.

    Note: Colab doesn't support URL parameters for env vars, so we link
    to a reproducer notebook that's generated with the model embedded.
    """
    # Basic Colab link to the notebook
    base_url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/notebooks/{notebook}"
    return base_url


def generate_reproducer_notebook(
    scenario: str,
    model: str,
    error_details: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a Jupyter notebook for reproducing a failure.

    Returns notebook as dict (can be saved as .ipynb).
    """
    notebook_map = {
        "basic_trace": "test_basic_trace.ipynb",
        "generation": "test_generation.ipynb",
        "hidden_states": "test_hidden_states.ipynb",
    }
    original_notebook = notebook_map.get(scenario, f"test_{scenario}.ipynb")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# NDIF Monitor - Reproducer for {model}\n",
                f"\n",
                f"**Scenario:** {scenario}\n",
                f"\n",
                f"This notebook reproduces a failure detected by NDIF Monitor.\n",
                f"Run all cells to see the issue.\n",
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Install dependencies\n",
                "!pip install -q nnsight torch\n",
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Set the model to test\n",
                f'MODEL_NAME = "{model}"\n',
                "print(f'Testing model: {MODEL_NAME}')\n",
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Set up nnsight\n",
                "import nnsight\n",
                "from nnsight import LanguageModel\n",
                "\n",
                "# Configure API key (get from nnsight.net)\n",
                "# nnsight.CONFIG.API.APIKEY = 'your-api-key'\n",
                "\n",
                f'model = LanguageModel(MODEL_NAME, device_map="auto")\n',
            ],
            "execution_count": None,
            "outputs": []
        },
    ]

    # Add scenario-specific test code
    if scenario == "basic_trace":
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Basic trace test\n",
                'prompt = "The quick brown fox"\n',
                "\n",
                "with model.trace(prompt, remote=True):\n",
                "    # Try to access hidden states\n",
                "    if hasattr(model, 'transformer'):\n",
                "        hidden = model.transformer.h[0].output[0].save()\n",
                "    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):\n",
                "        hidden = model.model.layers[0].output[0].save()\n",
                "    elif hasattr(model, 'gpt_neox'):\n",
                "        hidden = model.gpt_neox.layers[0].output[0].save()\n",
                "\n",
                "print(f'Hidden state shape: {hidden.shape}')\n",
                "print('SUCCESS: Basic trace works!')\n",
            ],
            "execution_count": None,
            "outputs": []
        })
    elif scenario == "generation":
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Generation test\n",
                'prompt = "Once upon a time"\n',
                "\n",
                "with model.generate(prompt, max_new_tokens=20, remote=True):\n",
                "    output = model.generator.output.save()\n",
                "\n",
                "generated = model.tokenizer.decode(output[0])\n",
                "print(f'Generated: {generated}')\n",
                "print('SUCCESS: Generation works!')\n",
            ],
            "execution_count": None,
            "outputs": []
        })
    elif scenario == "hidden_states":
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Hidden states extraction test\n",
                'prompt = "Hello world"\n',
                "\n",
                "with model.trace(prompt, remote=True):\n",
                "    if hasattr(model, 'transformer'):\n",
                "        layers = model.transformer.h\n",
                "    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):\n",
                "        layers = model.model.layers\n",
                "    elif hasattr(model, 'gpt_neox'):\n",
                "        layers = model.gpt_neox.layers\n",
                "    \n",
                "    states = [layer.output[0].save() for layer in layers]\n",
                "\n",
                "print(f'Extracted {len(states)} layer states')\n",
                "for i, s in enumerate(states[:3]):\n",
                "    print(f'  Layer {i}: {s.shape}')\n",
                "print('SUCCESS: Hidden states extraction works!')\n",
            ],
            "execution_count": None,
            "outputs": []
        })

    # Add error details if available
    if error_details:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Original Error\n",
                "\n",
                "```\n",
                f"{error_details}\n",
                "```\n",
            ]
        })

    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": f"NDIF_Reproducer_{scenario}_{model.replace('/', '_')}.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}
        },
        "cells": cells
    }


def generate_dashboard_html(
    history: HistoryStore,
    results_dir: Path,
    days: int = 365,
    github_repo: str = GITHUB_REPO,
) -> str:
    """Generate the dashboard HTML page.

    Args:
        history: HistoryStore with historical data
        results_dir: Directory containing model status files
        days: Number of days of history to show
        github_repo: GitHub repo for Colab links

    Returns:
        HTML string for the dashboard
    """
    # Load data
    daily_summary = history.get_daily_summary(days=days)
    recent_failures = history.get_recent_failures(days=7, limit=10)

    # Load current model statuses
    model_statuses = []
    for path in results_dir.glob("*.json"):
        if path.name.startswith(".") or path.name.startswith("run_") or path.name == "dashboard_data.json":
            continue
        status = ModelStatus.load(str(path))
        if status:
            model_statuses.append(status)
    model_statuses.sort(key=lambda s: s.model)

    # Get all models that have ever been tested
    all_models = set()
    for date_data in daily_summary.values():
        all_models.update(date_data.keys())
    all_models = sorted(all_models)

    # Generate date range for last N days
    today = datetime.utcnow().date()
    dates = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]

    # Convert data to JSON for JavaScript
    dashboard_data = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "host": get_hostname(),
        "user": get_username(),
        "days": days,
        "dates": dates,
        "models": all_models,
        "daily": daily_summary,
        "current": [s.to_dict() for s in model_statuses],
        "failures": [
            {
                "timestamp": f.timestamp,
                "model": f.model,
                "scenario": f.scenario,
                "status": f.status,
                "error_category": f.error_category,
                "details": f.details,
            }
            for f in recent_failures
        ],
        "github_repo": github_repo,
    }

    # Save data to data/ subdirectory
    data_dir = results_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "status.json"
    with open(data_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    return _generate_html(dashboard_data)


def _generate_html(data: Dict[str, Any]) -> str:
    """Generate the HTML content."""
    # Use a simpler approach - no f-string escaping needed
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NDIF Monitor</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔬</text></svg>">
    <style>
        :root {
            --ok: #10b981;
            --slow: #f59e0b;
            --degraded: #f97316;
            --failed: #ef4444;
            --unavailable: #6b7280;
            --cold: #4b5563;
            --bg: #0c0c0c;
            --card: #171717;
            --card-hover: #1f1f1f;
            --text: #fafafa;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --border: #27272a;
            --accent: #3b82f6;
            --stale-bg: #1a0a0a;
            --stale-border: #7f1d1d;
        }

        /* Stale data warning */
        body.stale {
            background: var(--stale-bg);
        }
        .stale-banner {
            display: none;
            background: linear-gradient(90deg, #7f1d1d 0%, #991b1b 50%, #7f1d1d 100%);
            color: #fecaca;
            padding: 0.75rem 1rem;
            text-align: center;
            font-weight: 500;
            border-bottom: 1px solid #dc2626;
        }
        body.stale .stale-banner {
            display: block;
        }
        .stale-banner strong {
            color: #fff;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }
        a { color: var(--accent); text-decoration: none; }
        a:hover { text-decoration: underline; }

        /* Layout */
        .container { max-width: 1200px; margin: 0 auto; padding: 0 1.5rem; }

        /* Header */
        header {
            border-bottom: 1px solid var(--border);
            padding: 1.5rem 0;
            margin-bottom: 2rem;
        }
        .header-inner {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 1.5rem;
        }
        .header-title h1 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .header-title p {
            color: var(--text-muted);
            font-size: 0.875rem;
        }
        .header-meta {
            text-align: right;
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        .header-meta .version {
            color: var(--text-secondary);
            font-family: ui-monospace, monospace;
        }

        /* Summary Stats */
        .summary {
            display: flex;
            gap: 2rem;
            padding: 1.5rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }
        .stat {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
        }
        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            line-height: 1;
        }
        .stat-value.ok { color: var(--ok); }
        .stat-value.slow { color: var(--slow); }
        .stat-value.failed { color: var(--failed); }
        .stat-value.total { color: var(--text-secondary); }
        .stat-label {
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Section */
        section { margin-bottom: 3rem; }
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .section-header h2 {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Legend */
        .legend {
            display: flex;
            gap: 1rem;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 2px;
        }
        .legend-dot.ok { background: var(--ok); }
        .legend-dot.slow { background: var(--slow); }
        .legend-dot.failed { background: var(--failed); }
        .legend-dot.unavailable { background: var(--unavailable); }
        .legend-dot.cold { background: var(--cold); }
        .legend-dot.empty { background: var(--border); }

        /* Calendar */
        .calendar-wrapper {
            overflow-x: auto;
            padding: 10px;  /* Space for hover effects */
            margin: -10px;  /* Compensate for padding */
            margin-bottom: 0;
            /* Scroll to show recent dates (right side) by default */
            direction: rtl;
        }
        .calendar-container {
            direction: ltr;
            width: max-content;
            margin: 0 auto;
        }
        .calendar-months {
            display: flex;
            margin-bottom: 4px;
            font-size: 0.7rem;
            color: var(--text-muted);
        }
        .calendar-month {
            text-align: left;
            white-space: nowrap;
        }
        .calendar {
            display: flex;
        }
        .calendar-week {
            display: flex;
            flex-direction: column;
        }
        .calendar-day {
            display: flex;
            flex-direction: row;
            cursor: pointer;
        }
        .calendar-segment {
            background: var(--border);
            transition: transform 0.15s, box-shadow 0.15s;
        }
        .calendar-day:hover {
            transform: scale(1.5);
            box-shadow: 0 0 0 2px var(--bg), 0 0 0 3px var(--text-muted);
            z-index: 10;
            position: relative;
        }
        .calendar-segment.ok { background: var(--ok); }
        .calendar-segment.slow { background: var(--slow); }
        .calendar-segment.degraded { background: var(--degraded); }
        .calendar-segment.failed { background: var(--failed); }
        .calendar-segment.unavailable { background: var(--unavailable); }
        .calendar-segment.cold { background: var(--cold); }

        /* Resize handle */
        .calendar-resize {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px 0 4px;
            cursor: ns-resize;
            user-select: none;
        }
        .resize-handle {
            width: 40px;
            height: 4px;
            background: var(--border);
            border-radius: 2px;
            transition: background 0.15s;
        }
        .calendar-resize:hover .resize-handle { background: var(--text-muted); }
        .resize-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-left: 8px;
        }

        .model-filter select {
            background: var(--card);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 0.4rem 0.75rem;
            border-radius: 0.375rem;
            font-size: 0.8rem;
            cursor: pointer;
        }
        .model-filter select:hover { border-color: var(--text-muted); }

        /* Model Grid */
        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1rem;
        }
        .model-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1rem 1.25rem;
            transition: border-color 0.15s;
        }
        .model-card:hover { border-color: var(--text-muted); }
        .model-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.75rem;
            gap: 0.5rem;
        }
        .model-name {
            font-weight: 600;
            font-size: 0.9rem;
            word-break: break-word;
        }
        .model-name .org {
            color: var(--text-muted);
            font-weight: 400;
        }
        .status-badge {
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            white-space: nowrap;
        }
        .status-badge.ok { background: var(--ok); color: #000; }
        .status-badge.slow { background: var(--slow); color: #000; }
        .status-badge.degraded { background: var(--degraded); color: #000; }
        .status-badge.failed { background: var(--failed); color: #fff; }
        .status-badge.unavailable { background: var(--unavailable); color: #fff; }
        .status-badge.cold { background: var(--cold); color: #fff; }

        .model-scenarios { margin-top: 0.75rem; }
        .scenario-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.4rem 0;
            border-top: 1px solid var(--border);
            font-size: 0.8rem;
        }
        .scenario-name-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .scenario-label {
            color: var(--text-secondary);
        }
        .scenario-colab {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-decoration: none;
            padding: 0.1rem 0.3rem;
            border: 1px solid var(--border);
            border-radius: 3px;
        }
        .scenario-colab:hover {
            color: var(--accent);
            border-color: var(--accent);
            text-decoration: none;
        }
        .scenario-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .scenario-time {
            color: var(--text-muted);
            font-size: 0.75rem;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .status-dot.ok { background: var(--ok); }
        .status-dot.slow { background: var(--slow); }
        .status-dot.degraded { background: var(--degraded); }
        .status-dot.failed { background: var(--failed); }
        .status-dot.unavailable { background: var(--unavailable); }
        .status-dot.cold { background: var(--cold); }

        .model-footer {
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
        }
        .model-footer .updated { color: var(--text-muted); }
        .colab-link {
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        /* Failures Table */
        .failures-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        .failures-table th {
            text-align: left;
            padding: 0.75rem 1rem;
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border);
        }
        .failures-table td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            vertical-align: top;
        }
        .failures-table tr:hover td { background: var(--card); }
        .error-details {
            color: var(--text-muted);
            font-size: 0.75rem;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .no-failures {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
        }

        /* Tooltip */
        .tooltip {
            position: fixed;
            background: var(--card);
            border: 1px solid var(--border);
            padding: 0.5rem 0.75rem;
            border-radius: 0.375rem;
            font-size: 0.75rem;
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            max-width: 280px;
        }
        .tooltip strong { color: var(--text); }
        .tooltip .tip-status { margin-top: 0.25rem; }

        /* Footer */
        footer {
            border-top: 1px solid var(--border);
            padding: 1.5rem 0;
            margin-top: 2rem;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        footer a { color: var(--text-secondary); }

        /* Loading */
        .loading {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
        }
        .loading-spinner {
            width: 24px;
            height: 24px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        @media (max-width: 640px) {
            .container { padding: 0 1rem; }
            .summary { gap: 1.5rem; }
            .stat-value { font-size: 2rem; }
            .model-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="stale-banner" id="staleBanner">
        <strong>Warning:</strong> Monitor data is stale. The monitoring system may be down.
        <span id="staleTime"></span>
    </div>
    <header>
        <div class="container">
            <div class="header-inner">
                <div class="header-title">
                    <h1>NDIF Monitor</h1>
                    <p>End-to-end testing of <a href="https://nnsight.net" target="_blank">nnsight</a> + <a href="https://ndif.us" target="_blank">NDIF</a></p>
                </div>
                <div class="header-meta">
                    <div>Updated: <span id="updated">-</span></div>
                    <div>nnsight <span class="version" id="version">-</span></div>
                    <div id="hostInfo"></div>
                </div>
            </div>
        </div>
    </header>

    <main class="container">
        <div class="summary" id="summary">
            <div class="stat"><span class="stat-value total" id="statTotal">-</span><span class="stat-label">Models</span></div>
            <div class="stat"><span class="stat-value ok" id="statOk">-</span><span class="stat-label">OK</span></div>
            <div class="stat"><span class="stat-value slow" id="statSlow">-</span><span class="stat-label">Slow</span></div>
            <div class="stat"><span class="stat-value failed" id="statFailed">-</span><span class="stat-label">Failed</span></div>
        </div>

        <section>
            <div class="section-header">
                <h2>Status History</h2>
                <div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap">
                    <div class="legend">
                        <div class="legend-item"><div class="legend-dot ok"></div>OK</div>
                        <div class="legend-item"><div class="legend-dot slow"></div>Slow</div>
                        <div class="legend-item"><div class="legend-dot failed"></div>Failed</div>
                        <div class="legend-item"><div class="legend-dot cold"></div>Cold</div>
                        <div class="legend-item"><div class="legend-dot empty"></div>No data</div>
                    </div>
                    <div class="model-filter">
                        <select id="modelSelect">
                            <option value="__all__">All models</option>
                        </select>
                    </div>
                </div>
            </div>
            <div class="calendar-wrapper">
                <div class="calendar-container">
                    <div class="calendar-months" id="calendarMonths"></div>
                    <div class="calendar" id="calendar"></div>
                </div>
            </div>
            <div class="calendar-resize" id="calendarResize">
                <div class="resize-handle"></div>
                <span class="resize-label" id="resizeLabel">6h segments</span>
            </div>
        </section>

        <section>
            <div class="section-header">
                <h2>Current Status</h2>
            </div>
            <div class="model-grid" id="modelGrid"></div>
        </section>

        <section>
            <div class="section-header">
                <h2>Recent Failures</h2>
            </div>
            <table class="failures-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Model</th>
                        <th>Test</th>
                        <th>Error</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody id="failuresBody"></tbody>
            </table>
        </section>
    </main>

    <footer>
        <div class="container">
            <a href="https://github.com/davidbau/ndif-monitor" target="_blank">GitHub</a> ·
            <a href="https://nnsight.net/documentation" target="_blank">nnsight docs</a> ·
            <a href="https://ndif.us" target="_blank">NDIF</a>
        </div>
    </footer>

    <div class="tooltip" id="tooltip" style="display:none"></div>
    <div class="loading" id="loading">
        <div class="loading-spinner"></div>
        Loading dashboard data...
    </div>

    <script>
        let DATA = null;
        let granularity = 4;  // Number of segments to show per day
        const GRANULARITY_LEVELS = [1, 2, 3, 4, 6, 8, 12, 24];
        const GRANULARITY_LABELS = {1: 'Daily', 2: '12h', 3: '8h', 4: '6h', 6: '4h', 8: '3h', 12: '2h', 24: '1h'};

        // Threshold for SLOW status (ms) - determined at analysis time
        const SLOW_THRESHOLD_MS = 35000;  // 35 seconds

        // Compute effective status, applying SLOW threshold to passing results
        // Treats stored SLOW as OK since SLOW is now determined only by duration
        function getEffectiveStatus(status, scenario, duration_ms) {
            // Failed/unavailable statuses pass through unchanged
            if (status === 'FAILED' || status === 'UNAVAILABLE' || status === 'COLD' || status === 'DEGRADED') {
                return status;
            }
            // For passing tests (OK or legacy SLOW), apply threshold
            if (!duration_ms) return 'OK';
            return duration_ms > SLOW_THRESHOLD_MS ? 'SLOW' : 'OK';
        }

        // Compute overall status from scenario statuses
        function computeOverallStatus(scenarios) {
            const statuses = Object.entries(scenarios || {}).map(([k, v]) =>
                getEffectiveStatus(v.status, k, v.duration_ms)
            );
            if (statuses.includes('FAILED') || statuses.includes('UNAVAILABLE')) return 'FAILED';
            if (statuses.includes('DEGRADED')) return 'DEGRADED';
            if (statuses.includes('SLOW')) return 'SLOW';
            if (statuses.every(s => s === 'COLD')) return 'COLD';
            return statuses.length ? 'OK' : 'UNKNOWN';
        }

        async function loadData() {
            try {
                const res = await fetch('data/status.json');
                if (!res.ok) throw new Error('Failed to load');
                DATA = await res.json();
                document.getElementById('loading').style.display = 'none';
                render();
                setupResizeHandle();
            } catch (e) {
                document.getElementById('loading').innerHTML =
                    '<div style="color:var(--failed)">Error loading data</div><div style="margin-top:0.5rem">' + e.message + '</div>';
            }
        }

        // Check if data is stale (older than 1 hour)
        const STALE_THRESHOLD_MS = 60 * 60 * 1000;  // 1 hour

        function checkStale() {
            if (!DATA || !DATA.generated) return;
            const generated = new Date(DATA.generated);
            const age = Date.now() - generated.getTime();
            if (age > STALE_THRESHOLD_MS) {
                document.body.classList.add('stale');
                const hours = Math.floor(age / (60 * 60 * 1000));
                const mins = Math.floor((age % (60 * 60 * 1000)) / (60 * 1000));
                let ageStr = hours > 0 ? hours + 'h ' + mins + 'm' : mins + 'm';
                document.getElementById('staleTime').textContent = '(Last update: ' + ageStr + ' ago)';
            } else {
                document.body.classList.remove('stale');
            }
        }

        function render() {
            // Check for stale data
            checkStale();

            // Header info
            document.getElementById('updated').textContent = formatTime(DATA.generated);
            if (DATA.nnsight_version) {
                document.getElementById('version').textContent = 'v' + DATA.nnsight_version;
            }
            if (DATA.host) {
                const info = DATA.user ? DATA.user + '@' + DATA.host : DATA.host;
                document.getElementById('hostInfo').textContent = info;
            }

            // Summary stats (compute SLOW status from thresholds)
            const stats = {total: 0, ok: 0, slow: 0, failed: 0};
            DATA.current.forEach(m => {
                stats.total++;
                const s = computeOverallStatus(m.scenarios).toLowerCase();
                if (s === 'ok') stats.ok++;
                else if (s === 'slow') stats.slow++;
                else if (s === 'failed' || s === 'unavailable' || s === 'degraded') stats.failed++;
            });
            document.getElementById('statTotal').textContent = stats.total;
            document.getElementById('statOk').textContent = stats.ok;
            document.getElementById('statSlow').textContent = stats.slow;
            document.getElementById('statFailed').textContent = stats.failed;

            // Model selector
            const select = document.getElementById('modelSelect');
            DATA.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m.split('/').pop();
                select.appendChild(opt);
            });
            select.onchange = () => renderCalendar(select.value);

            renderCalendar('__all__');
            renderModels();
            renderFailures();
        }

        function setupResizeHandle() {
            const handle = document.getElementById('calendarResize');
            let startY = 0;
            let startGranularity = granularity;

            // Calculate calendar height for a given granularity
            function calcHeight(g) {
                const segSize = Math.min(Math.max(window.innerWidth * 0.003, 3), 4);
                const dayGap = 1;
                const dayHeight = segSize * g + dayGap;
                return dayHeight * 7;  // 7 days per week column
            }

            function onMove(e) {
                const dy = (e.clientY || e.touches[0].clientY) - startY;
                // Target height = start height + drag distance
                const startHeight = calcHeight(startGranularity);
                const targetHeight = startHeight + dy;

                // Find the granularity level closest to target height
                let bestLevel = startGranularity;
                let bestDiff = Infinity;
                GRANULARITY_LEVELS.forEach(level => {
                    const diff = Math.abs(calcHeight(level) - targetHeight);
                    if (diff < bestDiff) {
                        bestDiff = diff;
                        bestLevel = level;
                    }
                });

                if (bestLevel !== granularity) {
                    granularity = bestLevel;
                    updateResizeLabel();
                    renderCalendar(document.getElementById('modelSelect').value);
                }
            }

            function onEnd() {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onEnd);
                document.removeEventListener('touchmove', onMove);
                document.removeEventListener('touchend', onEnd);
            }

            handle.addEventListener('mousedown', e => {
                startY = e.clientY;
                startGranularity = granularity;
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onEnd);
            });
            handle.addEventListener('touchstart', e => {
                startY = e.touches[0].clientY;
                startGranularity = granularity;
                document.addEventListener('touchmove', onMove);
                document.addEventListener('touchend', onEnd);
            });
        }

        function updateResizeLabel() {
            document.getElementById('resizeLabel').textContent = GRANULARITY_LABELS[granularity] + ' segments';
        }

        // Get segment statuses for a date and model, aggregated to current granularity
        function getSegmentStatuses(date, model) {
            if (!DATA.daily[date]) return Array(granularity).fill(null);

            // Map from 24 hourly slots to current granularity
            const hoursPerSegment = 24 / granularity;
            const result = [];

            for (let i = 0; i < granularity; i++) {
                const bucketStatuses = [];
                for (let j = 0; j < hoursPerSegment; j++) {
                    const hour = Math.floor(i * hoursPerSegment + j);
                    if (model === '__all__') {
                        // Aggregate across all models
                        Object.values(DATA.daily[date]).forEach(m => {
                            if (m.hours && m.hours[hour]) {
                                bucketStatuses.push(m.hours[hour]);
                            }
                        });
                    } else {
                        const d = DATA.daily[date][model];
                        if (d && d.hours && d.hours[hour]) {
                            bucketStatuses.push(d.hours[hour]);
                        }
                    }
                }
                // Compute worst status for this bucket
                if (bucketStatuses.length === 0) result.push(null);
                else if (bucketStatuses.includes('FAILED') || bucketStatuses.includes('UNAVAILABLE')) result.push('FAILED');
                else if (bucketStatuses.includes('DEGRADED')) result.push('DEGRADED');
                else if (bucketStatuses.includes('SLOW')) result.push('SLOW');
                else if (bucketStatuses.every(s => s === 'COLD')) result.push('COLD');
                else result.push('OK');
            }
            return result;
        }

        // Get overall status for a date (for tooltip)
        function getStatus(date, model) {
            if (!DATA.daily[date]) return null;
            if (model === '__all__') {
                const statuses = Object.values(DATA.daily[date]).map(m => m.status);
                if (statuses.includes('FAILED') || statuses.includes('UNAVAILABLE')) return 'FAILED';
                if (statuses.includes('DEGRADED')) return 'DEGRADED';
                if (statuses.includes('SLOW')) return 'SLOW';
                if (statuses.every(s => s === 'COLD')) return 'COLD';
                return statuses.length ? 'OK' : null;
            }
            const d = DATA.daily[date][model];
            return d ? d.status : null;
        }

        function renderCalendar(model) {
            const cal = document.getElementById('calendar');
            const monthsEl = document.getElementById('calendarMonths');
            cal.innerHTML = '';
            monthsEl.innerHTML = '';
            updateResizeLabel();

            const weeks = [];
            let week = [];
            const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

            // Track month boundaries for labels
            const monthStarts = [];
            let currentMonth = null;

            DATA.dates.forEach((date, i) => {
                const d = new Date(date + 'T00:00:00Z');
                const month = d.getUTCMonth();
                const year = d.getUTCFullYear();

                if (currentMonth !== month) {
                    monthStarts.push({weekIndex: weeks.length, month, year});
                    currentMonth = month;
                }

                if (d.getUTCDay() === 0 && week.length) {
                    weeks.push(week);
                    week = [];
                }
                week.push(date);
            });
            if (week.length) weeks.push(week);

            // Compute sizes based on granularity
            const segSize = Math.min(Math.max(window.innerWidth * 0.003, 3), 4);
            const dayGap = 1;
            const weekGap = Math.min(Math.max(window.innerWidth * 0.003, 2), 4);
            const dayWidth = segSize * granularity + dayGap;
            const weekWidth = dayWidth + weekGap;

            // Apply CSS custom properties
            cal.style.gap = weekGap + 'px';
            document.querySelectorAll('.calendar-week').forEach(w => w.style.gap = dayGap + 'px');

            // Render month labels
            monthStarts.forEach((ms, i) => {
                const label = document.createElement('span');
                label.className = 'calendar-month';
                const nextStart = (i + 1 < monthStarts.length) ? monthStarts[i + 1].weekIndex : weeks.length;
                const width = (nextStart - ms.weekIndex) * weekWidth;
                label.style.width = width + 'px';
                label.textContent = monthNames[ms.month];
                monthsEl.appendChild(label);
            });

            // Render weeks
            weeks.forEach(w => {
                const weekEl = document.createElement('div');
                weekEl.className = 'calendar-week';
                weekEl.style.gap = dayGap + 'px';

                // Pad first week
                const first = new Date(w[0] + 'T00:00:00Z').getUTCDay();
                for (let i = 0; i < first; i++) {
                    const pad = document.createElement('div');
                    pad.className = 'calendar-day';
                    pad.style.width = dayWidth + 'px';
                    pad.style.height = dayWidth + 'px';
                    weekEl.appendChild(pad);
                }

                w.forEach(date => {
                    const day = document.createElement('div');
                    day.className = 'calendar-day';
                    day.dataset.date = date;
                    day.style.gap = '0';

                    // Get segment statuses for this date
                    const segmentStatuses = getSegmentStatuses(date, model);

                    // Create segment elements
                    segmentStatuses.forEach((status, idx) => {
                        const seg = document.createElement('div');
                        seg.className = 'calendar-segment';
                        seg.style.width = segSize + 'px';
                        seg.style.height = dayWidth + 'px';
                        if (status) seg.classList.add(status.toLowerCase());
                        // Add data for segment-level tooltip
                        seg.dataset.date = date;
                        seg.dataset.segmentIdx = idx;
                        seg.onmouseenter = showSegmentTip;
                        seg.onmouseleave = hideTip;
                        day.appendChild(seg);
                    });

                    weekEl.appendChild(day);
                });
                cal.appendChild(weekEl);
            });
        }

        // Helper to get model folder name for Colab notebook paths
        function getModelFolder(model) {
            return model.replace('/', '--');
        }

        // Helper to build Colab URL for model-specific notebooks
        function getColabUrl(model, scenario) {
            const folder = getModelFolder(model);
            return 'https://colab.research.google.com/github/' + DATA.github_repo +
                   '/blob/main/notebooks/colab/' + folder + '/' + scenario + '.ipynb';
        }

        function renderModels() {
            const grid = document.getElementById('modelGrid');
            grid.innerHTML = '';

            DATA.current.forEach(m => {
                // Compute overall status from scenarios (applies SLOW thresholds)
                const overallStatus = computeOverallStatus(m.scenarios);
                const st = overallStatus.toLowerCase();
                const [org, name] = m.model.includes('/') ? m.model.split('/') : ['', m.model];

                let scenarios = '';
                Object.entries(m.scenarios || {}).forEach(([k, v]) => {
                    const dur = v.duration_ms ? (v.duration_ms / 1000).toFixed(1) + 's' : '';
                    // Apply SLOW threshold to each scenario
                    const effectiveStatus = getEffectiveStatus(v.status, k, v.duration_ms);
                    const scenarioColabUrl = getColabUrl(m.model, k);
                    scenarios += '<div class="scenario-row">' +
                        '<span class="scenario-name-group">' +
                        '<span class="scenario-label">' + k + '</span>' +
                        '<a href="' + scenarioColabUrl + '" target="_blank" class="scenario-colab" title="Run in Colab">colab</a>' +
                        '</span>' +
                        '<span class="scenario-status">' +
                        '<span class="scenario-time">' + dur + '</span>' +
                        '<span class="status-dot ' + effectiveStatus.toLowerCase() + '"></span>' +
                        '</span></div>';
                });

                const updated = m.last_updated ? formatTimeAgo(m.last_updated) : '-';

                const card = document.createElement('div');
                card.className = 'model-card';
                card.innerHTML =
                    '<div class="model-card-header">' +
                    '<div class="model-name">' + (org ? '<span class="org">' + org + '/</span>' : '') + name + '</div>' +
                    '<span class="status-badge ' + st + '">' + overallStatus + '</span>' +
                    '</div>' +
                    '<div class="model-scenarios">' + scenarios + '</div>' +
                    '<div class="model-footer">' +
                    '<span class="updated">Updated ' + updated + '</span>' +
                    '</div>';
                grid.appendChild(card);
            });
        }

        function renderFailures() {
            const tbody = document.getElementById('failuresBody');
            tbody.innerHTML = '';

            if (!DATA.failures.length) {
                tbody.innerHTML = '<tr><td colspan="5" class="no-failures">No recent failures - all tests passing!</td></tr>';
                return;
            }

            DATA.failures.forEach(f => {
                const colabUrl = getColabUrl(f.model, f.scenario);

                const tr = document.createElement('tr');
                tr.innerHTML =
                    '<td>' + formatTime(f.timestamp) + '</td>' +
                    '<td>' + f.model.split('/').pop() + '</td>' +
                    '<td>' + f.scenario + '</td>' +
                    '<td><span class="error-details" title="' + (f.details || '').replace(/"/g, '&quot;') + '">' + (f.error_category || f.status) + '</span></td>' +
                    '<td><a href="' + colabUrl + '" target="_blank" class="colab-link">Reproduce →</a></td>';
                tbody.appendChild(tr);
            });
        }

        function formatTime(iso) {
            const d = new Date(iso);
            return d.toLocaleDateString('en-US', {timeZone: 'America/New_York'}) + ' ' +
                   d.toLocaleTimeString('en-US', {timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit'});
        }

        function formatTimeAgo(iso) {
            const mins = Math.floor((Date.now() - new Date(iso)) / 60000);
            if (mins < 60) return mins + 'm ago';
            if (mins < 1440) return Math.floor(mins / 60) + 'h ago';
            return Math.floor(mins / 1440) + 'd ago';
        }

        const tooltip = document.getElementById('tooltip');

        // Format hour range for segment tooltip
        function getSegmentTimeRange(segmentIdx) {
            const hoursPerSegment = 24 / granularity;
            const startHour = segmentIdx * hoursPerSegment;
            const endHour = startHour + hoursPerSegment;
            const fmt = h => (h % 12 || 12) + (h < 12 ? 'am' : 'pm');
            return fmt(startHour) + '-' + fmt(endHour);
        }

        function showSegmentTip(e) {
            e.stopPropagation();
            const date = e.target.dataset.date;
            const segmentIdx = parseInt(e.target.dataset.segmentIdx);
            const model = document.getElementById('modelSelect').value;
            const timeRange = getSegmentTimeRange(segmentIdx);

            let html = '<strong>' + date + '</strong> <span style="color:var(--text-muted)">' + timeRange + '</span>';

            if (DATA.daily[date]) {
                const hoursPerSegment = 24 / granularity;
                const startHour = segmentIdx * hoursPerSegment;
                const endHour = startHour + hoursPerSegment;

                if (model === '__all__') {
                    // Aggregate all models for this time segment
                    const modelStatuses = {};
                    Object.entries(DATA.daily[date]).forEach(([m, d]) => {
                        if (d.hours) {
                            for (let h = startHour; h < endHour; h++) {
                                if (d.hours[h]) {
                                    const shortName = m.split('/').pop();
                                    if (!modelStatuses[shortName]) modelStatuses[shortName] = [];
                                    modelStatuses[shortName].push(d.hours[h]);
                                }
                            }
                        }
                    });
                    const entries = Object.entries(modelStatuses).slice(0, 6);
                    if (entries.length === 0) {
                        html += '<div class="tip-status">No tests in this period</div>';
                    } else {
                        entries.forEach(([m, statuses]) => {
                            const worst = statuses.includes('FAILED') ? 'FAILED' :
                                         statuses.includes('SLOW') ? 'SLOW' : 'OK';
                            html += '<div class="tip-status">' + m + ': ' + worst + '</div>';
                        });
                        if (Object.keys(modelStatuses).length > 6) html += '<div class="tip-status">...</div>';
                    }
                } else {
                    const d = DATA.daily[date][model];
                    if (d && d.hours) {
                        const hourStatuses = [];
                        for (let h = startHour; h < endHour; h++) {
                            if (d.hours[h]) hourStatuses.push(d.hours[h]);
                        }
                        if (hourStatuses.length === 0) {
                            html += '<div class="tip-status">No tests in this period</div>';
                        } else {
                            const worst = hourStatuses.includes('FAILED') ? 'FAILED' :
                                         hourStatuses.includes('SLOW') ? 'SLOW' : 'OK';
                            html += '<div class="tip-status">Status: ' + worst + '</div>';
                        }
                    } else {
                        html += '<div class="tip-status">No tests in this period</div>';
                    }
                }
            } else {
                html += '<div class="tip-status">No tests run</div>';
            }

            tooltip.innerHTML = html;
            tooltip.style.display = 'block';
            tooltip.style.left = Math.min(e.clientX + 12, window.innerWidth - 300) + 'px';
            tooltip.style.top = (e.clientY + 12) + 'px';
        }

        function showTip(e) {
            const date = e.target.dataset.date;
            const model = document.getElementById('modelSelect').value;
            let html = '<strong>' + date + '</strong>';

            if (DATA.daily[date]) {
                if (model === '__all__') {
                    const entries = Object.entries(DATA.daily[date]).slice(0, 6);
                    entries.forEach(([m, d]) => {
                        html += '<div class="tip-status">' + m.split('/').pop() + ': ' + d.status + '</div>';
                    });
                    if (Object.keys(DATA.daily[date]).length > 6) html += '<div class="tip-status">...</div>';
                } else {
                    const d = DATA.daily[date][model];
                    if (d) {
                        html += '<div class="tip-status">Status: ' + d.status + '</div>';
                        if (d.scenarios) {
                            Object.entries(d.scenarios).forEach(([k, v]) => {
                                html += '<div class="tip-status">' + k + ': ' + v + '</div>';
                            });
                        }
                    }
                }
            } else {
                html += '<div class="tip-status">No tests run</div>';
            }

            tooltip.innerHTML = html;
            tooltip.style.display = 'block';
            tooltip.style.left = Math.min(e.clientX + 12, window.innerWidth - 300) + 'px';
            tooltip.style.top = (e.clientY + 12) + 'px';
        }

        function hideTip() { tooltip.style.display = 'none'; }

        loadData();
    </script>
</body>
</html>"""
    return html


def generate_dashboard(
    results_dir: str = "results",
    output_file: str = "dashboard.html",
    days: int = 365,
    github_repo: str = GITHUB_REPO,
) -> str:
    """Generate dashboard HTML file.

    Args:
        results_dir: Directory containing results and history
        output_file: Output filename (relative to results_dir)
        days: Days of history to show
        github_repo: GitHub repo for Colab links

    Returns:
        Path to generated dashboard
    """
    results_path = Path(results_dir)
    history = HistoryStore(results_path / "history.jsonl")

    html = generate_dashboard_html(
        history=history,
        results_dir=results_path,
        days=days,
        github_repo=github_repo,
    )

    output_path = results_path / output_file
    with open(output_path, "w") as f:
        f.write(html)

    return str(output_path)
