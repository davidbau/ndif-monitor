"""Git synchronization for auto-generated notebooks.

Handles pushing newly generated Colab notebooks to GitHub so that
the dashboard's "Reproduce in Colab" links work correctly.
"""

import subprocess
import os
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def run_git_command(
    args: List[str],
    cwd: Optional[Path] = None,
    check: bool = True,
) -> Tuple[bool, str, str]:
    """Run a git command and return success status and output.

    Args:
        args: Git command arguments (without 'git' prefix)
        cwd: Working directory (defaults to current)
        check: If False, don't raise on non-zero exit

    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        success = result.returncode == 0
        if not success and check:
            logger.warning(f"Git command failed: git {' '.join(args)}")
            logger.warning(f"stderr: {result.stderr}")
        return success, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        logger.error(f"Git command timed out: git {' '.join(args)}")
        return False, "", "Command timed out"
    except Exception as e:
        logger.error(f"Git command error: {e}")
        return False, "", str(e)


def get_repo_root(path: Optional[Path] = None) -> Optional[Path]:
    """Get the root directory of the git repository."""
    success, stdout, _ = run_git_command(
        ["rev-parse", "--show-toplevel"],
        cwd=path,
        check=False,
    )
    if success and stdout:
        return Path(stdout)
    return None


def is_file_tracked(file_path: Path, repo_root: Optional[Path] = None) -> bool:
    """Check if a file is already tracked in git.

    Args:
        file_path: Path to the file (absolute or relative to repo)
        repo_root: Repository root directory

    Returns:
        True if file is tracked in git
    """
    if repo_root is None:
        repo_root = get_repo_root(file_path.parent)
    if repo_root is None:
        return False

    # Make path relative to repo root
    try:
        rel_path = file_path.relative_to(repo_root)
    except ValueError:
        rel_path = file_path

    # Check if file is in git index
    success, _, _ = run_git_command(
        ["ls-files", "--error-unmatch", str(rel_path)],
        cwd=repo_root,
        check=False,
    )
    return success


def is_notebook_dir_tracked(model_folder: str, notebooks_base: Path) -> bool:
    """Check if a model's notebook directory is already tracked.

    Args:
        model_folder: Model folder name (e.g., "Qwen--Qwen2.5-Coder-7B-Instruct")
        notebooks_base: Base path to notebooks/colab directory

    Returns:
        True if any notebooks for this model are tracked
    """
    model_dir = notebooks_base / model_folder
    if not model_dir.exists():
        return False

    # Check if any .ipynb files in this dir are tracked
    for notebook in model_dir.glob("*.ipynb"):
        if is_file_tracked(notebook):
            return True
    return False


def sync_notebooks_to_github(
    model_name: str,
    notebooks_base: Path,
    commit_message: Optional[str] = None,
) -> bool:
    """Commit and push newly generated notebooks for a model to GitHub.

    Only commits if the notebooks are not already tracked. This ensures we
    don't create unnecessary commits for existing notebooks.

    Args:
        model_name: Full model name (e.g., "Qwen/Qwen2.5-Coder-7B-Instruct")
        notebooks_base: Path to notebooks/colab directory
        commit_message: Optional custom commit message

    Returns:
        True if notebooks were pushed (or already existed), False on error
    """
    model_folder = model_name.replace("/", "--")
    model_dir = notebooks_base / model_folder

    if not model_dir.exists():
        logger.warning(f"Notebook directory does not exist: {model_dir}")
        return False

    repo_root = get_repo_root(notebooks_base)
    if repo_root is None:
        logger.warning("Not in a git repository, skipping sync")
        return False

    # Check if notebooks are already tracked
    if is_notebook_dir_tracked(model_folder, notebooks_base):
        logger.debug(f"Notebooks for {model_name} already in git, skipping")
        return True

    # Get relative path for git commands
    try:
        rel_model_dir = model_dir.relative_to(repo_root)
    except ValueError:
        logger.error(f"Model dir {model_dir} not under repo root {repo_root}")
        return False

    logger.info(f"Syncing new notebooks for {model_name} to GitHub")

    # Stage the new notebooks
    success, _, stderr = run_git_command(
        ["add", str(rel_model_dir)],
        cwd=repo_root,
        check=False,
    )
    if not success:
        logger.error(f"Failed to stage notebooks: {stderr}")
        return False

    # Check if there's anything to commit
    success, stdout, _ = run_git_command(
        ["diff", "--cached", "--name-only"],
        cwd=repo_root,
        check=False,
    )
    if not stdout.strip():
        logger.debug("No changes to commit")
        return True

    # Commit
    if commit_message is None:
        commit_message = f"Add Colab notebooks for {model_name}\n\nAuto-generated by NDIF Monitor"

    success, _, stderr = run_git_command(
        ["commit", "-m", commit_message],
        cwd=repo_root,
        check=False,
    )
    if not success:
        logger.error(f"Failed to commit: {stderr}")
        return False

    # Pull and push with retry for concurrent updates from multiple monitors
    max_retries = 3
    for attempt in range(max_retries):
        # Pull first to avoid conflicts (rebase to keep our commit on top)
        success, _, stderr = run_git_command(
            ["pull", "--rebase"],
            cwd=repo_root,
            check=False,
        )
        if not success:
            logger.warning(f"Pull failed (may be offline or no remote): {stderr}")
            # Continue anyway - push might still work if we're up to date

        # Push
        success, _, stderr = run_git_command(
            ["push"],
            cwd=repo_root,
            check=False,
        )
        if success:
            break

        if attempt < max_retries - 1:
            logger.warning(f"Push failed (attempt {attempt + 1}), retrying: {stderr}")
            import time
            time.sleep(1)  # Brief delay before retry
        else:
            logger.error(f"Failed to push after {max_retries} attempts: {stderr}")
            return False

    logger.info(f"Successfully pushed notebooks for {model_name}")
    return True


def sync_all_new_notebooks(notebooks_base: Path) -> Tuple[int, int]:
    """Sync all untracked notebook directories to GitHub.

    Finds all model directories that have notebooks but aren't tracked,
    and commits/pushes them in a single commit.

    Args:
        notebooks_base: Path to notebooks/colab directory

    Returns:
        Tuple of (synced_count, error_count)
    """
    if not notebooks_base.exists():
        return 0, 0

    # Resolve to absolute path for consistent comparison
    notebooks_base = notebooks_base.resolve()

    repo_root = get_repo_root(notebooks_base)
    if repo_root is None:
        logger.warning("Not in a git repository")
        return 0, 1

    # Find all model directories with untracked notebooks
    untracked_models = []
    for model_dir in notebooks_base.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith("."):
            continue

        # Check if this model has any notebooks
        notebooks = list(model_dir.glob("*.ipynb"))
        if not notebooks:
            continue

        # Check if already tracked
        if not is_notebook_dir_tracked(model_dir.name, notebooks_base):
            untracked_models.append(model_dir.name)

    if not untracked_models:
        logger.debug("No new notebooks to sync")
        return 0, 0

    logger.info(f"Found {len(untracked_models)} models with untracked notebooks")

    # Stage all untracked model directories
    try:
        rel_base = notebooks_base.relative_to(repo_root)
    except ValueError:
        logger.error(f"Notebooks dir {notebooks_base} not under repo root {repo_root}")
        return 0, len(untracked_models)

    staged_models = []
    for model_folder in untracked_models:
        rel_path = rel_base / model_folder
        success, _, _ = run_git_command(
            ["add", str(rel_path)],
            cwd=repo_root,
            check=False,
        )
        if success:
            staged_models.append(model_folder)

    if not staged_models:
        return 0, len(untracked_models)

    # Create commit message
    if len(staged_models) == 1:
        model_name = staged_models[0].replace("--", "/")
        commit_message = f"Add Colab notebooks for {model_name}\n\nAuto-generated by NDIF Monitor"
    else:
        model_list = "\n".join(f"- {m.replace('--', '/')}" for m in staged_models)
        commit_message = f"Add Colab notebooks for {len(staged_models)} models\n\n{model_list}\n\nAuto-generated by NDIF Monitor"

    # Commit
    success, _, stderr = run_git_command(
        ["commit", "-m", commit_message],
        cwd=repo_root,
        check=False,
    )
    if not success:
        logger.error(f"Failed to commit: {stderr}")
        return 0, len(untracked_models)

    # Pull and push with retry for concurrent updates
    max_retries = 3
    for attempt in range(max_retries):
        # Pull first to get any remote changes
        success, _, stderr = run_git_command(
            ["pull", "--rebase"],
            cwd=repo_root,
            check=False,
        )
        if not success:
            logger.warning(f"Pull failed (attempt {attempt + 1}): {stderr}")

        # Push
        success, _, stderr = run_git_command(
            ["push"],
            cwd=repo_root,
            check=False,
        )
        if success:
            break

        if attempt < max_retries - 1:
            logger.warning(f"Push failed (attempt {attempt + 1}), retrying: {stderr}")
            import time
            time.sleep(1)  # Brief delay before retry
        else:
            logger.error(f"Failed to push after {max_retries} attempts: {stderr}")
            return 0, len(untracked_models)

    logger.info(f"Successfully pushed notebooks for {len(staged_models)} models")
    return len(staged_models), len(untracked_models) - len(staged_models)
