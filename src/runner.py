"""Main test runner for NDIF Monitor."""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

from .models import (
    ModelInfo,
    get_available_models,
    get_test_models,
    select_test_models,
    fetch_ndif_status,
    print_status_summary,
    BASELINE_MODELS,
)
from .results import (
    MonitorRun,
    TestResult,
    Status,
    ModelStatus,
    ScenarioResult,
    model_to_filename,
)
from .jupyter_executor import VenvManager, run_notebook_test
from .history import HistoryStore, HistoryEntry, get_hostname, get_username
from .notebook_generator import generate_colab_notebooks_for_model


@dataclass
class Scenario:
    """Test scenario configuration."""
    name: str                    # Scenario identifier (also notebook name without .ipynb)
    description: str             # Human-readable description
    timeout: int = 300           # Execution timeout in seconds
    model_specific: bool = False # If True, only run on specific architectures
    architectures: List[str] = field(default_factory=list)  # If model_specific


# Default test scenarios - notebooks are in notebooks/colab/{model}/{scenario}.ipynb
DEFAULT_SCENARIOS = [
    Scenario(
        name="basic_trace",
        description="Basic model.trace() with hidden state extraction",
        timeout=90,  # 90s should be plenty for basic trace
    ),
    Scenario(
        name="generation",
        description="Text generation with model.generate()",
        timeout=90,
    ),
    Scenario(
        name="hidden_states",
        description="Extract hidden states from all layers",
        timeout=120,  # Hidden states may take longer
    ),
]


class CycleState:
    """Tracks round-robin cycling state across runs."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.last_index = -1
        self._load()

    def _load(self):
        """Load state from file."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                data = json.load(f)
                self.last_index = data.get("last_index", -1)

    def _save(self):
        """Save state to file."""
        with open(self.state_file, "w") as f:
            json.dump({"last_index": self.last_index}, f)

    def next_model(self, models: List[ModelInfo]) -> ModelInfo:
        """Get next model in round-robin order."""
        if not models:
            raise ValueError("No models available")
        self.last_index = (self.last_index + 1) % len(models)
        self._save()
        return models[self.last_index]


class MonitorRunner:
    """Orchestrates NDIF monitoring test runs."""

    def __init__(
        self,
        notebooks_dir: str,
        results_dir: str,
        scenarios: Optional[List[Scenario]] = None,
    ):
        """Initialize monitor runner.

        Args:
            notebooks_dir: Directory containing test notebooks
            results_dir: Directory for storing results JSON files
            scenarios: List of scenarios to run (uses defaults if None)
        """
        self.notebooks_dir = Path(notebooks_dir)
        self.results_dir = Path(results_dir)
        self.scenarios = scenarios or DEFAULT_SCENARIOS

        # Ensure directories exist
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Colab notebooks directory
        self.colab_dir = self.notebooks_dir / "colab"
        self.colab_dir.mkdir(parents=True, exist_ok=True)

        # Cycle state for round-robin
        self.cycle_state = CycleState(self.results_dir / ".cycle_state.json")

        # History store for dashboard
        self.history = HistoryStore(self.results_dir / "history.jsonl")

    def get_notebook_path(self, model_name: str, scenario: Scenario) -> Path:
        """Get full path to a model's scenario notebook.

        Colab notebooks are stored at: notebooks/colab/{model_filename}/{scenario}.ipynb
        """
        model_filename = model_name.replace("/", "--")
        return self.colab_dir / model_filename / f"{scenario.name}.ipynb"

    def ensure_notebooks_generated(self, model_name: str) -> List[Path]:
        """Ensure Colab notebooks exist for a model, generating if needed.

        Returns list of generated/existing notebook paths.
        """
        scenario_names = [s.name for s in self.scenarios]
        return generate_colab_notebooks_for_model(
            model_name=model_name,
            output_dir=self.colab_dir,
            scenarios=scenario_names,
        )

    def get_model_status_path(self, model_name: str) -> Path:
        """Get path to model's status JSON file."""
        return self.results_dir / model_to_filename(model_name)

    def load_model_status(self, model_name: str) -> Optional[ModelStatus]:
        """Load existing model status from file."""
        path = self.get_model_status_path(model_name)
        return ModelStatus.load(str(path))

    def save_model_status(self, status: ModelStatus) -> str:
        """Save model status to file."""
        path = self.get_model_status_path(status.model)
        status.save(str(path))
        return str(path)

    def update_model_status(
        self,
        model_name: str,
        scenario_name: str,
        result: TestResult,
        nnsight_version: str,
    ) -> ModelStatus:
        """Update model status with new test result.

        Loads existing status, updates the scenario, and saves.
        """
        now = datetime.utcnow()

        # Load existing or create new
        status = self.load_model_status(model_name)
        if status is None:
            status = ModelStatus(
                model=model_name,
                last_updated=now,
                nnsight_version=nnsight_version,
                scenarios={},
            )

        # Get existing scenario result to preserve last_success
        existing = status.scenarios.get(scenario_name)
        last_success = existing.last_success if existing else None

        # Update last_success if this test passed
        if result.status in (Status.OK, Status.SLOW):
            last_success = now

        # Create new scenario result
        status.scenarios[scenario_name] = ScenarioResult(
            status=result.status,
            duration_ms=result.duration_ms,
            last_checked=now,
            last_success=last_success,
            error_category=result.error_category,
            details=result.details,
        )

        # Update metadata
        status.last_updated = now
        status.nnsight_version = nnsight_version

        # Save immediately
        self.save_model_status(status)

        # Record to history
        history_entry = HistoryEntry(
            timestamp=now.isoformat() + "Z",
            model=model_name,
            scenario=scenario_name,
            status=result.status.value,
            duration_ms=result.duration_ms,
            error_category=result.error_category.value if result.error_category else None,
            details=result.details,
            host=get_hostname(),
            user=get_username(),
        )
        self.history.append(history_entry)

        return status

    def run_single_model(
        self,
        model: ModelInfo,
        venv: VenvManager,
        nnsight_version: str,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> List[TestResult]:
        """Run all scenarios for a single model."""
        results = []

        # Check if model is intentionally offline (COLD in NDIF)
        if not model.is_available:
            print(f" (cold)")
            for scenario in self.scenarios:
                if scenario.model_specific:
                    if model.architecture.value not in scenario.architectures:
                        continue
                result = TestResult(
                    model=model.model_key,
                    scenario=scenario.name,
                    status=Status.COLD,
                    duration_ms=0,
                    details=f"Model is {model.deployment_level.value} (intentionally offline)",
                )
                results.append(result)
                self.update_model_status(
                    model.model_key,
                    scenario.name,
                    result,
                    nnsight_version,
                )
            return results

        # Generate Colab notebooks for this model (they have hardcoded model names)
        self.ensure_notebooks_generated(model.model_key)

        for scenario in self.scenarios:
            # Skip if model_specific and architecture doesn't match
            if scenario.model_specific:
                if model.architecture.value not in scenario.architectures:
                    continue

            print(f"\n  {scenario.name}...", end=" ", flush=True)

            notebook_path = self.get_notebook_path(model.model_key, scenario)
            if not notebook_path.exists():
                print(f"⚠ notebook not found")
                result = TestResult(
                    model=model.model_key,
                    scenario=scenario.name,
                    status=Status.FAILED,
                    duration_ms=0,
                    details=f"Notebook not found: {notebook_path.name}",
                )
            else:
                # Run the Colab notebook (has hardcoded model name, no MODEL_NAME env var needed)
                result = run_notebook_test(
                    notebook_path=str(notebook_path),
                    model_name=model.model_key,
                    scenario_name=scenario.name,
                    venv=venv,
                    timeout=scenario.timeout,
                    extra_env=env_vars,
                )

            results.append(result)

            # Update per-model status file immediately
            self.update_model_status(
                model.model_key,
                scenario.name,
                result,
                nnsight_version,
            )

            # Print result
            status_symbols = {
                Status.OK: "✓",
                Status.SLOW: "~",
                Status.DEGRADED: "⚠",
                Status.FAILED: "✗",
                Status.UNAVAILABLE: "·",
                Status.COLD: "○",
            }
            symbol = status_symbols.get(result.status, "?")
            duration_str = f"{result.duration_ms / 1000:.1f}s"
            print(f"{symbol} {result.status.value} ({duration_str})")
            if result.details and result.status == Status.FAILED:
                # Print first line of error
                first_line = result.details.split('\n')[0][:60]
                print(f"    {first_line}...")

        return results

    def run(
        self,
        models: Optional[List[ModelInfo]] = None,
        max_per_architecture: int = 2,
        env_vars: Optional[Dict[str, str]] = None,
        cycle: bool = False,
    ) -> MonitorRun:
        """Run the test matrix.

        Args:
            models: List of models to test (auto-selects from NDIF if None)
            max_per_architecture: Maximum models per architecture
            env_vars: Extra environment variables (e.g., NDIF_API, HF_TOKEN)
            cycle: If True, run only one model (round-robin across runs)

        Returns:
            MonitorRun with all test results
        """
        start_time = time.time()
        timestamp = datetime.utcnow()

        # Fetch available models if not provided
        if models is None:
            print("Fetching NDIF model status...")
            status = fetch_ndif_status()
            available = get_available_models(status, hot_only=True)
            print_status_summary(available)

            # Get baseline models + extra hot models
            models = get_test_models(
                status=status,
                include_extra_hot=True,
                max_extra_per_architecture=max_per_architecture,
            )

            # Show which are baseline vs extra
            baseline_keys = set(BASELINE_MODELS)
            baseline_count = sum(1 for m in models if m.model_key in baseline_keys)
            extra_count = len(models) - baseline_count
            print(f"\nBaseline models: {baseline_count}, Extra hot models: {extra_count}")

        # In cycle mode, pick just one model
        if cycle and models:
            model = self.cycle_state.next_model(models)
            models = [model]
            print(f"\nCycle mode: testing {model.model_key}")
        else:
            print(f"\nSelected {len(models)} models for testing")
            for m in models:
                print(f"  - {m.model_key}")

        # Use existing environment (conda or venv) instead of creating fresh venv each time
        # This is much faster (~3 min saved) and the conda env has the right Python version
        venv = VenvManager(use_system_python=True)
        results: List[TestResult] = []
        nnsight_version = "unknown"

        try:
            # Get nnsight version from current environment
            nnsight_version = venv.get_package_version("nnsight") or "unknown"
            print(f"nnsight version: {nnsight_version}")

            # Build environment for notebooks
            notebook_env = {}
            if env_vars:
                notebook_env.update(env_vars)

            # Add API keys from environment if not provided
            for key in ["NDIF_API", "HF_TOKEN"]:
                if key not in notebook_env and os.environ.get(key):
                    notebook_env[key] = os.environ[key]

            # nnsight expects NDIF_API_KEY (not NDIF_API)
            if "NDIF_API" in notebook_env and "NDIF_API_KEY" not in notebook_env:
                notebook_env["NDIF_API_KEY"] = notebook_env["NDIF_API"]

            # Run tests for each model
            print(f"\nRunning {len(models)} model(s) × {len(self.scenarios)} scenarios")
            print("=" * 60)

            for model in models:
                print(f"\n[{model.short_name}]")
                model_results = self.run_single_model(
                    model=model,
                    venv=venv,
                    nnsight_version=nnsight_version,
                    env_vars=notebook_env,
                )
                results.extend(model_results)

        finally:
            # Always cleanup venv
            venv.cleanup()

        # Build final result
        duration_seconds = time.time() - start_time
        run_result = MonitorRun(
            timestamp=timestamp,
            nnsight_version=nnsight_version,
            duration_seconds=duration_seconds,
            tests=results,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        summary = run_result.summary
        print(f"Total: {summary['total']} | OK: {summary['ok']} | SLOW: {summary['slow']} | "
              f"FAILED: {summary['failed']} | UNAVAILABLE: {summary['unavailable']}")
        print(f"Duration: {duration_seconds:.1f}s")

        # Show updated model status files
        print(f"\nModel status files updated in: {self.results_dir}/")
        for model in models:
            status = self.load_model_status(model.model_key)
            if status:
                symbol = {"OK": "✓", "SLOW": "~", "FAILED": "✗", "UNAVAILABLE": "·"}.get(
                    status.overall_status.value, "?"
                )
                print(f"  {symbol} {model_to_filename(model.model_key)}")

        return run_result

    def save_result(self, run: MonitorRun, filename: Optional[str] = None) -> str:
        """Save run results to JSON file (legacy format).

        Args:
            run: MonitorRun to save
            filename: Custom filename (auto-generates if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"run_{run.timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        output_path = self.results_dir / filename
        run.save(str(output_path))
        print(f"\nRun log saved to: {output_path}")
        return str(output_path)

    def list_model_statuses(self) -> List[ModelStatus]:
        """List all model status files."""
        statuses = []
        for path in self.results_dir.glob("*.json"):
            if path.name.startswith(".") or path.name.startswith("run_"):
                continue
            status = ModelStatus.load(str(path))
            if status:
                statuses.append(status)
        return sorted(statuses, key=lambda s: s.model)

    def print_all_statuses(self):
        """Print summary of all tracked model statuses."""
        statuses = self.list_model_statuses()
        if not statuses:
            print("No model status files found.")
            return

        print(f"\nTracked Models: {len(statuses)}")
        print("-" * 70)

        for status in statuses:
            symbol = {"OK": "✓", "SLOW": "~", "FAILED": "✗", "UNAVAILABLE": "·"}.get(
                status.overall_status.value, "?"
            )
            age = datetime.utcnow() - status.last_updated
            age_str = f"{age.total_seconds() / 3600:.1f}h ago" if age.total_seconds() < 86400 else f"{age.days}d ago"

            print(f"{symbol} {status.model:<45} {status.overall_status.value:<6} ({age_str})")

            # Show scenario details
            for name, scenario in status.scenarios.items():
                s_symbol = {"OK": "✓", "SLOW": "~", "FAILED": "✗", "UNAVAILABLE": "·"}.get(
                    scenario.status.value, "?"
                )
                last_ok = ""
                if scenario.last_success:
                    ok_age = datetime.utcnow() - scenario.last_success
                    if ok_age.total_seconds() < 3600:
                        last_ok = f"last OK {ok_age.total_seconds() / 60:.0f}m ago"
                    elif ok_age.total_seconds() < 86400:
                        last_ok = f"last OK {ok_age.total_seconds() / 3600:.1f}h ago"
                    else:
                        last_ok = f"last OK {ok_age.days}d ago"
                print(f"    {s_symbol} {name}: {scenario.status.value} {last_ok}")


def run_monitor(
    notebooks_dir: str = "notebooks",
    results_dir: str = "results",
    max_models: int = 2,
    env_vars: Optional[Dict[str, str]] = None,
    save_results: bool = True,
    cycle: bool = False,
) -> MonitorRun:
    """Convenience function to run the monitoring suite.

    Args:
        notebooks_dir: Directory containing test notebooks
        results_dir: Directory for results
        max_models: Maximum models per architecture to test
        env_vars: Extra environment variables
        save_results: Whether to save run log (per-model files always saved)
        cycle: Run one model at a time, cycling through

    Returns:
        MonitorRun with all results
    """
    runner = MonitorRunner(
        notebooks_dir=notebooks_dir,
        results_dir=results_dir,
    )

    result = runner.run(
        max_per_architecture=max_models,
        env_vars=env_vars,
        cycle=cycle,
    )

    if save_results:
        runner.save_result(result)

    return result
