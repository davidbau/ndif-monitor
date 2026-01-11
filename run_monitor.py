#!/usr/bin/env python3
"""NDIF Monitor - CLI entry point.

Run end-to-end tests of NDIF and nnsight infrastructure.

Usage:
    python run_monitor.py [OPTIONS]

Examples:
    # Run full test suite
    python run_monitor.py

    # Test only 1 model per architecture
    python run_monitor.py --max-models 1

    # Show available models without testing
    python run_monitor.py --status-only

    # Round-robin: test one model per run, cycling through all
    python run_monitor.py --cycle

    # Show all tracked model statuses
    python run_monitor.py --show-status

Cloud Deployment:
    For AWS/GCP, use their secrets management:
    - AWS: AWS Secrets Manager or SSM Parameter Store
    - GCP: Google Secret Manager
    - GitHub Actions: Repository secrets
    See README.md for detailed setup instructions.
"""

import argparse
import os
import sys
import re
from pathlib import Path


def load_env_local():
    """Load environment variables from .env.local file."""
    env_file = Path(__file__).parent / ".env.local"
    if not env_file.exists():
        return

    print(f"Loading credentials from {env_file.name}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse KEY="value" or KEY=value
            match = re.match(r'^([A-Z_]+)=["\']?([^"\']+)["\']?$', line)
            if match and not os.environ.get(match.group(1)):
                os.environ[match.group(1)] = match.group(2)


# Load .env.local before anything else
load_env_local()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import get_available_models, print_status_summary, fetch_ndif_status, BASELINE_MODELS
from src.runner import run_monitor, MonitorRunner
from src.dashboard import generate_dashboard
from src.notebook_generator import generate_all_colab_notebooks

import shutil
import subprocess


def generate_and_commit_colab_notebooks(
    notebooks_dir: Path,
    models: list,
    github_repo: str,
) -> None:
    """Generate Colab notebooks for all models and commit them.

    Args:
        notebooks_dir: Base notebooks directory
        models: List of model names to generate notebooks for
        github_repo: GitHub repo name for notebook metadata
    """
    colab_dir = notebooks_dir / "colab"

    print(f"\nGenerating Colab notebooks for {len(models)} models...")
    results = generate_all_colab_notebooks(
        models=models,
        output_dir=colab_dir,
        scenarios=["basic_trace", "generation", "hidden_states"],
    )

    # Count total notebooks generated
    total = sum(len(paths) for paths in results.values())
    print(f"\nGenerated {total} Colab notebooks in {colab_dir}/")


def deploy_dashboard(results_dir: Path, deploy_path: str, notebooks_dir: Path = None) -> None:
    """Deploy dashboard files to target directory.

    Copies:
    - index.html (dashboard)
    - data/status.json (dashboard data)
    - data/models/*.json (per-model status files)
    - notebooks/colab/* (Colab notebooks for reproducibility)
    """
    deploy_dir = Path(deploy_path)

    # Create deploy directories
    deploy_dir.mkdir(parents=True, exist_ok=True)
    (deploy_dir / "data").mkdir(exist_ok=True)
    (deploy_dir / "data" / "models").mkdir(exist_ok=True)

    # Copy dashboard HTML as index.html
    dashboard_src = results_dir / "dashboard.html"
    if dashboard_src.exists():
        shutil.copy2(dashboard_src, deploy_dir / "index.html")
        print(f"  Deployed: index.html")

    # Copy data/status.json
    data_src = results_dir / "data" / "status.json"
    if data_src.exists():
        shutil.copy2(data_src, deploy_dir / "data" / "status.json")
        print(f"  Deployed: data/status.json")

    # Copy model status JSON files to data/models/
    model_files = list(results_dir.glob("*.json"))
    model_count = 0
    for src in model_files:
        if src.name.startswith(".") or src.name.startswith("run_"):
            continue
        dst = deploy_dir / "data" / "models" / src.name
        shutil.copy2(src, dst)
        model_count += 1

    if model_count > 0:
        print(f"  Deployed: {model_count} model status files to data/models/")

    # Copy Colab notebooks if available
    if notebooks_dir:
        colab_src = notebooks_dir / "colab"
        if colab_src.exists():
            colab_dst = deploy_dir / "notebooks"
            if colab_dst.exists():
                shutil.rmtree(colab_dst)
            shutil.copytree(colab_src, colab_dst)
            # Count notebooks
            nb_count = len(list(colab_dst.rglob("*.ipynb")))
            print(f"  Deployed: {nb_count} Colab notebooks to notebooks/")

    # Set permissions for web serving (chmod a+rX)
    try:
        subprocess.run(["chmod", "-R", "a+rX", str(deploy_dir)], check=True)
        print(f"  Set permissions: chmod -R a+rX {deploy_dir}")
    except subprocess.CalledProcessError:
        print(f"  Warning: Could not set permissions on {deploy_dir}")

    print(f"\nDashboard deployed to: {deploy_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="NDIF Monitor - End-to-end testing of NDIF and nnsight",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  NDIF_API    API key for NDIF (from nnsight.net)
  HF_TOKEN    HuggingFace token (for gated models like Llama)

Examples:
  # Run full test suite
  python run_monitor.py

  # Quick test with minimal models
  python run_monitor.py --max-models 1

  # Just check NDIF status
  python run_monitor.py --status-only

  # Round-robin testing (one model per run)
  python run_monitor.py --cycle

  # Show all tracked model statuses
  python run_monitor.py --show-status
        """,
    )

    parser.add_argument(
        "--max-models",
        type=int,
        default=2,
        help="Maximum models to test per architecture (default: 2)",
    )

    parser.add_argument(
        "--notebooks-dir",
        type=str,
        default="notebooks",
        help="Directory containing test notebooks (default: notebooks)",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for storing results (default: results)",
    )

    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only show NDIF status, don't run tests",
    )

    parser.add_argument(
        "--show-status",
        action="store_true",
        help="Show all tracked model statuses from previous runs",
    )

    parser.add_argument(
        "--cycle",
        action="store_true",
        help="Round-robin mode: test one model per run, cycling through all",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save run log file (per-model files are always saved)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Custom output filename for run log",
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate dashboard HTML after tests (auto-enabled with --deploy)",
    )

    parser.add_argument(
        "--deploy",
        type=str,
        metavar="PATH",
        help="Deploy dashboard to specified directory (e.g., /share/projects/ndif-monitor/www)",
    )

    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Only regenerate dashboard from existing history (no tests)",
    )

    parser.add_argument(
        "--github-repo",
        type=str,
        default="davidbau/ndif-monitor",
        help="GitHub repo for Colab links (default: davidbau/ndif-monitor)",
    )

    parser.add_argument(
        "--generate-notebooks",
        action="store_true",
        help="Generate Colab notebooks for all tracked models",
    )

    args = parser.parse_args()

    # Set up results directory
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).parent / results_dir

    # Show tracked statuses
    if args.show_status:
        runner = MonitorRunner(
            notebooks_dir=args.notebooks_dir,
            results_dir=str(results_dir),
        )
        runner.print_all_statuses()
        return 0

    # Set up notebooks directory early (needed for several modes)
    notebooks_dir = Path(args.notebooks_dir)
    if not notebooks_dir.is_absolute():
        notebooks_dir = Path(__file__).parent / notebooks_dir

    # Generate notebooks mode
    if args.generate_notebooks:
        print("Generating Colab notebooks for all baseline models...")
        generate_and_commit_colab_notebooks(
            notebooks_dir=notebooks_dir,
            models=BASELINE_MODELS,
            github_repo=args.github_repo,
        )
        return 0

    # Dashboard only mode
    if args.dashboard_only:
        print("Generating dashboard from existing history...")
        dashboard_path = generate_dashboard(
            results_dir=str(results_dir),
            github_repo=args.github_repo,
        )
        print(f"Dashboard generated: {dashboard_path}")

        if args.deploy:
            deploy_dashboard(results_dir, args.deploy, notebooks_dir)

        return 0

    # Check for API credentials
    ndif_api = os.environ.get("NDIF_API")
    hf_token = os.environ.get("HF_TOKEN")

    if not ndif_api:
        print("Warning: NDIF_API environment variable not set")
        print("Tests may fail without valid API credentials")
        print("Get your key at https://nnsight.net\n")

    # Fetch and display status
    print("NDIF Monitor")
    print("=" * 60)

    try:
        status = fetch_ndif_status()
        models = get_available_models(status, hot_only=True)
        print_status_summary(models)
    except Exception as e:
        print(f"Error fetching NDIF status: {e}")
        if args.status_only:
            return 1
        print("Continuing with test execution...\n")

    if args.status_only:
        return 0

    # Check for notebooks
    if not notebooks_dir.exists():
        print(f"\nError: Notebooks directory not found: {notebooks_dir}")
        print("Create test notebooks first.")
        return 1

    notebook_files = list(notebooks_dir.glob("test_*.ipynb"))
    if not notebook_files:
        print(f"\nError: No test notebooks found in {notebooks_dir}")
        print("Expected notebooks like test_basic_trace.ipynb")
        return 1

    print(f"\nFound {len(notebook_files)} test notebooks:")
    for nb in notebook_files:
        print(f"  - {nb.name}")

    # Run tests
    print("\n" + "=" * 60)
    print("Starting test run...")
    print("=" * 60)

    env_vars = {}
    if ndif_api:
        env_vars["NDIF_API"] = ndif_api
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    runner = MonitorRunner(
        notebooks_dir=str(notebooks_dir),
        results_dir=str(results_dir),
    )

    result = runner.run(
        max_per_architecture=args.max_models,
        env_vars=env_vars if env_vars else None,
        cycle=args.cycle,
    )

    # Save run log (per-model files are always saved during run)
    if not args.no_save:
        runner.save_result(result, args.output)

    # Generate dashboard if requested or deploying
    if args.dashboard or args.deploy:
        print("\n" + "=" * 60)
        print("Generating dashboard...")
        dashboard_path = generate_dashboard(
            results_dir=str(results_dir),
            github_repo=args.github_repo,
        )
        print(f"Dashboard generated: {dashboard_path}")

        if args.deploy:
            print("\nDeploying dashboard...")
            deploy_dashboard(results_dir, args.deploy, notebooks_dir)

    # Return exit code based on failures
    summary = result.summary
    if summary["failed"] > 0 or summary["unavailable"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
