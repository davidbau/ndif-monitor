"""Jupyter notebook execution with fresh virtual environments."""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime


def get_random_ports(count: int = 5) -> List[int]:
    """Get random high ports for Jupyter kernel communication."""
    # Use ports in the dynamic/private range (49152-65535)
    return [random.randint(49152, 65000) for _ in range(count)]

from .results import TestResult, Status, ErrorCategory, classify_error, determine_status


@dataclass
class ExecutionResult:
    """Result of notebook execution."""
    success: bool
    duration_ms: int
    output: str
    error: Optional[str] = None
    notebook_output: Optional[Dict[str, Any]] = None


class VenvManager:
    """Manages temporary virtual environments for isolated testing."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize venv manager.

        Args:
            base_dir: Base directory for venvs. Uses temp dir if None.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
        self.venv_path: Optional[Path] = None

    def create_venv(self, name: str = "ndif-test") -> Path:
        """Create a fresh virtual environment.

        Args:
            name: Name prefix for the venv directory

        Returns:
            Path to the venv directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.venv_path = self.base_dir / f"{name}_{timestamp}"

        print(f"Creating virtual environment at {self.venv_path}...")
        subprocess.run(
            [sys.executable, "-m", "venv", str(self.venv_path)],
            check=True,
            capture_output=True,
        )
        return self.venv_path

    def get_python(self) -> str:
        """Get path to Python executable in venv."""
        if not self.venv_path:
            raise RuntimeError("No venv created. Call create_venv() first.")

        if sys.platform == "win32":
            return str(self.venv_path / "Scripts" / "python.exe")
        return str(self.venv_path / "bin" / "python")

    def get_pip(self) -> str:
        """Get path to pip executable in venv."""
        if not self.venv_path:
            raise RuntimeError("No venv created. Call create_venv() first.")

        if sys.platform == "win32":
            return str(self.venv_path / "Scripts" / "pip.exe")
        return str(self.venv_path / "bin" / "pip")

    def install_packages(self, packages: List[str], quiet: bool = True) -> None:
        """Install packages in the venv.

        Args:
            packages: List of package specs (e.g., ["nnsight", "torch"])
            quiet: Suppress pip output
        """
        if not packages:
            return

        cmd = [self.get_pip(), "install"]
        if quiet:
            cmd.append("-q")
        cmd.extend(packages)

        print(f"Installing packages: {', '.join(packages)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Package installation failed: {result.stderr}")

    def get_package_version(self, package: str) -> Optional[str]:
        """Get installed version of a package."""
        result = subprocess.run(
            [self.get_pip(), "show", package],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.split("\n"):
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
        return None

    def cleanup(self) -> None:
        """Remove the virtual environment."""
        if self.venv_path and self.venv_path.exists():
            print(f"Cleaning up venv at {self.venv_path}")
            shutil.rmtree(self.venv_path, ignore_errors=True)
            self.venv_path = None


class NotebookExecutor:
    """Executes Jupyter notebooks and captures results."""

    def __init__(self, venv: VenvManager):
        """Initialize notebook executor.

        Args:
            venv: VenvManager instance with active venv
        """
        self.venv = venv

    def execute_notebook(
        self,
        notebook_path: str,
        env_vars: Optional[Dict[str, str]] = None,
        timeout: int = 300,
    ) -> ExecutionResult:
        """Execute a notebook and capture results.

        Args:
            notebook_path: Path to the .ipynb file
            env_vars: Environment variables to set (e.g., MODEL_NAME)
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with success status, output, timing
        """
        notebook_path = Path(notebook_path)
        if not notebook_path.exists():
            return ExecutionResult(
                success=False,
                duration_ms=0,
                output="",
                error=f"Notebook not found: {notebook_path}",
            )

        # Create temp directory for executed notebook output
        temp_dir = tempfile.mkdtemp(prefix="nbexec_")
        output_name = "executed_notebook"
        output_path = os.path.join(temp_dir, f"{output_name}.ipynb")

        # Build environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # Use isolated runtime directory to avoid port conflicts
        runtime_dir = os.path.join(temp_dir, "jupyter_runtime")
        os.makedirs(runtime_dir, exist_ok=True)
        env["JUPYTER_RUNTIME_DIR"] = runtime_dir

        # Build nbconvert command with random ports bound to localhost only
        # Use -c to apply nest_asyncio before running nbconvert (fixes Python 3.8 asyncio issues)
        ports = get_random_ports(5)
        cmd = [
            self.venv.get_python(),
            "-c",
            # nest_asyncio must be applied before importing anything that uses asyncio
            "import nest_asyncio; nest_asyncio.apply(); "
            "import sys; sys.argv = sys.argv[1:]; "  # Remove -c from argv
            "from nbconvert import nbconvertapp; nbconvertapp.main()",
            "--to", "notebook",
            "--execute",
            "--output", output_name,
            "--output-dir", temp_dir,
            "--ExecutePreprocessor.timeout", str(timeout),
            # Bind kernel ports to localhost only (not exposed to network)
            "--KernelManager.ip=127.0.0.1",
            f"--KernelManager.shell_port={ports[0]}",
            f"--KernelManager.iopub_port={ports[1]}",
            f"--KernelManager.stdin_port={ports[2]}",
            f"--KernelManager.hb_port={ports[3]}",
            f"--KernelManager.control_port={ports[4]}",
            str(notebook_path),
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout + 30,  # Extra buffer for nbconvert overhead
            )
            duration_ms = int((time.time() - start_time) * 1000)

            # Read executed notebook to get cell outputs
            notebook_output = None
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    notebook_output = json.load(f)

            if result.returncode == 0:
                return ExecutionResult(
                    success=True,
                    duration_ms=duration_ms,
                    output=result.stdout + result.stderr,
                    notebook_output=notebook_output,
                )
            else:
                return ExecutionResult(
                    success=False,
                    duration_ms=duration_ms,
                    output=result.stdout,
                    error=result.stderr,
                    notebook_output=notebook_output,
                )

        except subprocess.TimeoutExpired:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                duration_ms=duration_ms,
                output="",
                error=f"Notebook execution timed out after {timeout}s",
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                duration_ms=duration_ms,
                output="",
                error=str(e),
            )
        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def extract_cell_outputs(
        self,
        notebook_output: Optional[Dict[str, Any]],
    ) -> Tuple[str, Optional[str]]:
        """Extract text output and errors from executed notebook cells.

        Args:
            notebook_output: Parsed notebook JSON

        Returns:
            Tuple of (combined_output, error_message_if_any)
        """
        if not notebook_output:
            return "", None

        outputs = []
        errors = []

        for cell in notebook_output.get("cells", []):
            if cell.get("cell_type") != "code":
                continue

            for output in cell.get("outputs", []):
                output_type = output.get("output_type")

                if output_type == "stream":
                    text = output.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    outputs.append(text)

                elif output_type == "error":
                    ename = output.get("ename", "Error")
                    evalue = output.get("evalue", "")
                    traceback = output.get("traceback", [])
                    error_text = f"{ename}: {evalue}"
                    errors.append(error_text)
                    # Include traceback in outputs for debugging
                    outputs.append("\n".join(traceback) if traceback else error_text)

                elif output_type in ("execute_result", "display_data"):
                    data = output.get("data", {})
                    if "text/plain" in data:
                        text = data["text/plain"]
                        if isinstance(text, list):
                            text = "".join(text)
                        outputs.append(text)

        combined_output = "\n".join(outputs)
        error = errors[0] if errors else None

        return combined_output, error


def run_notebook_test(
    notebook_path: str,
    model_name: str,
    scenario_name: str,
    venv: VenvManager,
    timeout: int = 300,
    threshold_ms: int = 30000,
    extra_env: Optional[Dict[str, str]] = None,
) -> TestResult:
    """Run a single notebook test and return structured result.

    Args:
        notebook_path: Path to the notebook file
        model_name: Name of the model being tested
        scenario_name: Name of the test scenario
        venv: VenvManager with installed packages
        timeout: Execution timeout in seconds
        threshold_ms: Duration threshold for SLOW status
        extra_env: Additional environment variables

    Returns:
        TestResult with status, timing, and error classification
    """
    executor = NotebookExecutor(venv)

    # Set up environment with model name
    env_vars = {"MODEL_NAME": model_name}
    if extra_env:
        env_vars.update(extra_env)

    # Execute notebook
    result = executor.execute_notebook(
        notebook_path,
        env_vars=env_vars,
        timeout=timeout,
    )

    # Extract outputs
    output_text, cell_error = executor.extract_cell_outputs(result.notebook_output)

    # Determine error from cell error or execution error
    error_text = cell_error or result.error

    # Classify result
    status = determine_status(
        success=result.success and not cell_error,
        duration_ms=result.duration_ms,
        threshold_ms=threshold_ms,
        error_text=error_text,
    )

    error_category = None
    if not result.success or cell_error:
        error_category = classify_error(error_text or "Unknown error")

    return TestResult(
        model=model_name,
        scenario=scenario_name,
        status=status,
        duration_ms=result.duration_ms,
        error_category=error_category,
        details=error_text,
        output=output_text[:2000] if output_text else None,  # Truncate for storage
    )
