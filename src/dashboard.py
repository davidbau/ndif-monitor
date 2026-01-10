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
    """Generate the HTML content.

    The HTML is static and loads data from dashboard_data.json.
    """
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NDIF Monitor Dashboard</title>
    <style>
        :root {{
            --ok: #22c55e;
            --slow: #eab308;
            --degraded: #f97316;
            --failed: #ef4444;
            --unavailable: #6b7280;
            --empty: #1f2937;
            --bg: #0f172a;
            --card: #1e293b;
            --text: #f1f5f9;
            --text-dim: #94a3b8;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
            line-height: 1.5;
        }}
        h1, h2, h3 {{ font-weight: 600; margin-bottom: 1rem; }}
        h1 {{ font-size: 1.5rem; }}
        h2 {{ font-size: 1.25rem; color: var(--text-dim); }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        .timestamp {{ color: var(--text-dim); font-size: 0.875rem; }}
        .card {{
            background: var(--card);
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
        }}
        .legend {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}

        /* Calendar heatmap */
        .calendar-container {{
            overflow-x: auto;
            padding-bottom: 1rem;
        }}
        .calendar {{
            display: flex;
            gap: 2px;
        }}
        .calendar-week {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        .calendar-day {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
            cursor: pointer;
            transition: transform 0.1s;
        }}
        .calendar-day:hover {{
            transform: scale(1.5);
            z-index: 10;
        }}
        .calendar-day.empty {{ background: var(--empty); }}
        .calendar-day.ok {{ background: var(--ok); }}
        .calendar-day.slow {{ background: var(--slow); }}
        .calendar-day.degraded {{ background: var(--degraded); }}
        .calendar-day.failed {{ background: var(--failed); }}
        .calendar-day.unavailable {{ background: var(--unavailable); }}

        .month-labels {{
            display: flex;
            font-size: 0.75rem;
            color: var(--text-dim);
            margin-bottom: 0.5rem;
            padding-left: 2px;
        }}
        .month-label {{
            flex: 1;
            min-width: 50px;
        }}

        /* Model selector */
        .model-selector {{
            margin-bottom: 1rem;
        }}
        .model-selector select {{
            background: var(--bg);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
            cursor: pointer;
        }}

        /* Current status grid */
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
        }}
        .model-card {{
            background: var(--bg);
            border-radius: 0.5rem;
            padding: 1rem;
            border: 1px solid var(--border);
        }}
        .model-name {{
            font-weight: 600;
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
            word-break: break-all;
        }}
        .model-status {{
            display: inline-block;
            padding: 0.125rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .model-status.ok {{ background: var(--ok); color: #000; }}
        .model-status.slow {{ background: var(--slow); color: #000; }}
        .model-status.degraded {{ background: var(--degraded); color: #000; }}
        .model-status.failed {{ background: var(--failed); }}
        .model-status.unavailable {{ background: var(--unavailable); }}

        .scenarios {{
            margin-top: 0.75rem;
            font-size: 0.75rem;
        }}
        .scenario {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.25rem 0;
            border-bottom: 1px solid var(--border);
        }}
        .scenario:last-child {{ border-bottom: none; }}
        .scenario-status {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        .status-dot.ok {{ background: var(--ok); }}
        .status-dot.slow {{ background: var(--slow); }}
        .status-dot.degraded {{ background: var(--degraded); }}
        .status-dot.failed {{ background: var(--failed); }}
        .status-dot.unavailable {{ background: var(--unavailable); }}

        .last-success {{ color: var(--text-dim); font-size: 0.7rem; }}

        /* Failures table */
        .failures-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}
        .failures-table th, .failures-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        .failures-table th {{
            color: var(--text-dim);
            font-weight: 500;
        }}
        .colab-link {{
            color: #60a5fa;
            text-decoration: none;
        }}
        .colab-link:hover {{ text-decoration: underline; }}

        /* Tooltip */
        .tooltip {{
            position: fixed;
            background: var(--card);
            border: 1px solid var(--border);
            padding: 0.5rem 0.75rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            pointer-events: none;
            z-index: 100;
            max-width: 300px;
        }}

        @media (max-width: 640px) {{
            body {{ padding: 1rem; }}
            .calendar-day {{ width: 8px; height: 8px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>NDIF Monitor Dashboard</h1>
            <div class="timestamp">Last updated: <span id="updated"></span> <span id="host" style="opacity:0.6"></span></div>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background:var(--ok)"></div> OK</div>
            <div class="legend-item"><div class="legend-color" style="background:var(--slow)"></div> Slow</div>
            <div class="legend-item"><div class="legend-color" style="background:var(--degraded)"></div> Degraded</div>
            <div class="legend-item"><div class="legend-color" style="background:var(--failed)"></div> Failed</div>
            <div class="legend-item"><div class="legend-color" style="background:var(--unavailable)"></div> Unavailable</div>
            <div class="legend-item"><div class="legend-color" style="background:var(--empty)"></div> No data</div>
        </div>
    </div>

    <div class="card">
        <h2>Status History</h2>
        <div class="model-selector">
            <select id="modelSelect">
                <option value="__all__">All Models (worst status per day)</option>
            </select>
        </div>
        <div class="month-labels" id="monthLabels"></div>
        <div class="calendar-container">
            <div class="calendar" id="calendar"></div>
        </div>
    </div>

    <div class="card">
        <h2>Current Status</h2>
        <div class="status-grid" id="statusGrid"></div>
    </div>

    <div class="card">
        <h2>Recent Failures</h2>
        <table class="failures-table" id="failuresTable">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Model</th>
                    <th>Scenario</th>
                    <th>Error</th>
                    <th>Reproduce</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>

    <div class="tooltip" id="tooltip" style="display:none"></div>

    <div id="loading" style="text-align:center;padding:2rem;color:var(--text-dim)">Loading data...</div>

    <script>
        let DATA = null;

        // Load data from external JSON file
        async function loadData() {
            try {
                const response = await fetch('data/status.json');
                if (!response.ok) throw new Error('Failed to load data');
                DATA = await response.json();
                document.getElementById('loading').style.display = 'none';
                initDashboard();
            } catch (error) {
                document.getElementById('loading').innerHTML =
                    '<span style="color:var(--failed)">Error loading data: ' + error.message + '</span>';
            }
        }

        function initDashboard() {
            // Initialize
            document.getElementById('updated').textContent = new Date(DATA.generated).toLocaleString();
            if (DATA.host) {
                const hostInfo = DATA.user ? DATA.user + '@' + DATA.host : DATA.host;
                document.getElementById('host').textContent = '(' + hostInfo + ')';
            }

            // Populate model selector
            const modelSelect = document.getElementById('modelSelect');
            DATA.models.forEach(model => {
                const opt = document.createElement('option');
                opt.value = model;
                opt.textContent = model;
                modelSelect.appendChild(opt);
            });

            // Event listeners
            modelSelect.addEventListener('change', () => renderCalendar(modelSelect.value));

            // Initial render
            renderCalendar('__all__');
            renderStatusGrid();
            renderFailures();
        }

        // Get status for a date/model combination
        function getStatus(date, model) {
            if (!DATA.daily[date]) return null;
            if (model === '__all__') {{
                // Return worst status across all models
                const statuses = Object.values(DATA.daily[date]).map(m => m.status);
                if (statuses.includes('UNAVAILABLE')) return 'UNAVAILABLE';
                if (statuses.includes('FAILED')) return 'FAILED';
                if (statuses.includes('DEGRADED')) return 'DEGRADED';
                if (statuses.includes('SLOW')) return 'SLOW';
                if (statuses.length > 0) return 'OK';
                return null;
            }}
            const modelData = DATA.daily[date][model];
            return modelData ? modelData.status : null;
        }}

        // Render calendar
        function renderCalendar(selectedModel) {{
            const calendar = document.getElementById('calendar');
            const monthLabels = document.getElementById('monthLabels');
            calendar.innerHTML = '';
            monthLabels.innerHTML = '';

            // Group dates by week
            const weeks = [];
            let currentWeek = [];
            let currentMonth = '';
            const months = [];

            DATA.dates.forEach((date, i) => {{
                const d = new Date(date + 'T00:00:00Z');
                const dayOfWeek = d.getUTCDay();

                // Track months for labels
                const month = date.substring(0, 7);
                if (month !== currentMonth) {{
                    months.push({{ month, week: weeks.length }});
                    currentMonth = month;
                }}

                // Start new week on Sunday
                if (dayOfWeek === 0 && currentWeek.length > 0) {{
                    weeks.push(currentWeek);
                    currentWeek = [];
                }}

                currentWeek.push(date);
            }});
            if (currentWeek.length > 0) weeks.push(currentWeek);

            // Render month labels
            const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            months.forEach((m, i) => {{
                const label = document.createElement('span');
                label.className = 'month-label';
                const monthNum = parseInt(m.month.split('-')[1]) - 1;
                label.textContent = monthNames[monthNum];
                monthLabels.appendChild(label);
            }});

            // Render weeks
            weeks.forEach(week => {{
                const weekEl = document.createElement('div');
                weekEl.className = 'calendar-week';

                // Pad start of first week
                const firstDay = new Date(week[0] + 'T00:00:00Z').getUTCDay();
                for (let i = 0; i < firstDay; i++) {{
                    const pad = document.createElement('div');
                    pad.className = 'calendar-day empty';
                    weekEl.appendChild(pad);
                }}

                week.forEach(date => {{
                    const day = document.createElement('div');
                    day.className = 'calendar-day';

                    const status = getStatus(date, selectedModel);
                    if (status) {{
                        day.classList.add(status.toLowerCase());
                    }} else {{
                        day.classList.add('empty');
                    }}

                    day.dataset.date = date;
                    day.addEventListener('mouseenter', showTooltip);
                    day.addEventListener('mouseleave', hideTooltip);
                    weekEl.appendChild(day);
                }});

                calendar.appendChild(weekEl);
            }});
        }}

        // Tooltip handlers
        const tooltip = document.getElementById('tooltip');

        function showTooltip(e) {{
            const date = e.target.dataset.date;
            const model = modelSelect.value;

            let content = `<strong>${{date}}</strong><br>`;

            if (DATA.daily[date]) {{
                if (model === '__all__') {{
                    const models = Object.entries(DATA.daily[date]);
                    models.slice(0, 5).forEach(([m, data]) => {{
                        content += `${{m.split('/').pop()}}: ${{data.status}}<br>`;
                    }});
                    if (models.length > 5) content += `... and ${{models.length - 5}} more`;
                }} else {{
                    const data = DATA.daily[date][model];
                    if (data) {{
                        content += `Status: ${{data.status}}<br>`;
                        if (data.scenarios) {{
                            Object.entries(data.scenarios).forEach(([s, st]) => {{
                                content += `${{s}}: ${{st}}<br>`;
                            }});
                        }}
                    }} else {{
                        content += 'No data';
                    }}
                }}
            }} else {{
                content += 'No data';
            }}

            tooltip.innerHTML = content;
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 10) + 'px';
            tooltip.style.top = (e.clientY + 10) + 'px';
        }}

        function hideTooltip() {{
            tooltip.style.display = 'none';
        }}

        // Render current status grid
        function renderStatusGrid() {{
            const grid = document.getElementById('statusGrid');
            grid.innerHTML = '';

            DATA.current.forEach(model => {{
                const card = document.createElement('div');
                card.className = 'model-card';

                const statusClass = model.overall_status.toLowerCase();
                let lastOkStr = '';
                if (model.last_all_ok) {{
                    const lastOk = new Date(model.last_all_ok);
                    const ago = Math.floor((Date.now() - lastOk) / 1000 / 60);
                    if (ago < 60) lastOkStr = `${{ago}}m ago`;
                    else if (ago < 1440) lastOkStr = `${{Math.floor(ago/60)}}h ago`;
                    else lastOkStr = `${{Math.floor(ago/1440)}}d ago`;
                }}

                let scenariosHtml = '';
                Object.entries(model.scenarios || {{}}).forEach(([name, s]) => {{
                    const st = s.status.toLowerCase();
                    let lastSuccessStr = '';
                    if (s.last_success) {{
                        const ls = new Date(s.last_success);
                        const ago = Math.floor((Date.now() - ls) / 1000 / 60);
                        if (ago < 60) lastSuccessStr = `OK ${{ago}}m ago`;
                        else if (ago < 1440) lastSuccessStr = `OK ${{Math.floor(ago/60)}}h ago`;
                        else lastSuccessStr = `OK ${{Math.floor(ago/1440)}}d ago`;
                    }}
                    scenariosHtml += `
                        <div class="scenario">
                            <span>${{name}}</span>
                            <span class="scenario-status">
                                <span class="last-success">${{lastSuccessStr}}</span>
                                <span class="status-dot ${{st}}"></span>
                            </span>
                        </div>
                    `;
                }});

                // Colab link for testing this model
                const colabUrl = `https://colab.research.google.com/github/${{DATA.github_repo}}/blob/main/notebooks/test_basic_trace.ipynb`;

                card.innerHTML = `
                    <div class="model-name">${{model.model}}</div>
                    <span class="model-status ${{statusClass}}">${{model.overall_status}}</span>
                    ${{lastOkStr ? `<span class="last-success" style="margin-left:0.5rem">All OK ${{lastOkStr}}</span>` : ''}}
                    <a href="${{colabUrl}}" target="_blank" class="colab-link" style="float:right;font-size:0.75rem">Test in Colab</a>
                    <div class="scenarios">${{scenariosHtml}}</div>
                `;

                grid.appendChild(card);
            }});
        }}

        // Render failures table
        function renderFailures() {{
            const tbody = document.querySelector('#failuresTable tbody');
            tbody.innerHTML = '';

            if (DATA.failures.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-dim)">No recent failures</td></tr>';
                return;
            }}

            DATA.failures.forEach(f => {{
                const tr = document.createElement('tr');
                const time = new Date(f.timestamp).toLocaleString();
                const shortModel = f.model.split('/').pop();

                // Generate Colab link
                const notebookMap = {{
                    'basic_trace': 'test_basic_trace.ipynb',
                    'generation': 'test_generation.ipynb',
                    'hidden_states': 'test_hidden_states.ipynb'
                }};
                const notebook = notebookMap[f.scenario] || `test_${{f.scenario}}.ipynb`;
                const colabUrl = `https://colab.research.google.com/github/${{DATA.github_repo}}/blob/main/notebooks/${{notebook}}`;

                tr.innerHTML = `
                    <td>${{time}}</td>
                    <td title="${{f.model}}">${{shortModel}}</td>
                    <td>${{f.scenario}}</td>
                    <td title="${{f.details || ''}}">${{f.error_category || f.status}}</td>
                    <td><a href="${{colabUrl}}" target="_blank" class="colab-link">Open in Colab</a></td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        // Start loading data
        loadData();
    </script>
</body>
</html>
'''


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
