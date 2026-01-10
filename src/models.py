"""NDIF status API and model registry for NDIF Monitor."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
import requests
import json


NDIF_STATUS_URL = "https://api.ndif.us/status"


# Baseline models that are typically always hot and should always be tested.
# These represent core architectures that NDIF keeps running as dedicated deployments.
BASELINE_MODELS = [
    # Small/fast models
    "openai-community/gpt2",              # GPT-2: smallest, fastest canary
    "EleutherAI/gpt-j-6b",                # GPT-J: 6B params

    # 7-8B models
    "meta-llama/Llama-2-7b-hf",           # Llama 2 7B
    "meta-llama/Llama-3.1-8B",            # Llama 3.1 8B
    "allenai/Olmo-3-1025-7B",             # OLMo 7B (shared)

    # Large models (70B)
    "meta-llama/Llama-3.1-70B",           # Llama 3.1 70B base
    "meta-llama/Llama-3.1-70B-Instruct",  # Llama 3.1 70B instruct
    "meta-llama/Llama-3.3-70B-Instruct",  # Llama 3.3 70B instruct

    # Very large models (405B)
    "meta-llama/Llama-3.1-405B-Instruct", # Llama 3.1 405B instruct
]


class DeploymentLevel(Enum):
    """NDIF model deployment status."""
    HOT = "HOT"    # Actively running, immediately available
    COLD = "COLD"  # Offline, requires activation


class ModelArchitecture(Enum):
    """Model architecture families."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    GPT2 = "gpt2"
    GPTJ = "gptj"
    GPT_NEOX = "gpt_neox"
    GEMMA = "gemma"
    OLMO = "olmo"
    PHI = "phi"
    DEEPSEEK = "deepseek"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Information about an NDIF model deployment."""
    model_key: str            # Full key like "meta-llama/Llama-3.1-8B"
    repo_id: str              # HuggingFace repo ID
    deployment_level: DeploymentLevel
    application_state: str    # e.g., "RUNNING"
    n_params: Optional[int]   # Parameter count
    dedicated: bool           # Exclusive vs shared resources
    architecture: ModelArchitecture

    @property
    def is_available(self) -> bool:
        """Check if model is immediately usable."""
        return (
            self.deployment_level == DeploymentLevel.HOT
            and self.application_state == "RUNNING"
        )

    @property
    def short_name(self) -> str:
        """Get short model name without org prefix."""
        if "/" in self.model_key:
            return self.model_key.split("/")[-1]
        return self.model_key


def detect_architecture(model_key: str, config_str: Optional[str] = None) -> ModelArchitecture:
    """Detect model architecture from model key or config."""
    key_lower = model_key.lower()

    # Check model key patterns
    patterns = [
        (["llama", "llama-2", "llama-3"], ModelArchitecture.LLAMA),
        (["mistral"], ModelArchitecture.MISTRAL),
        (["qwen"], ModelArchitecture.QWEN),
        (["gpt2", "gpt-2"], ModelArchitecture.GPT2),
        (["gptj", "gpt-j"], ModelArchitecture.GPTJ),
        (["pythia", "gpt-neox"], ModelArchitecture.GPT_NEOX),
        (["gemma"], ModelArchitecture.GEMMA),
        (["olmo"], ModelArchitecture.OLMO),
        (["phi"], ModelArchitecture.PHI),
        (["deepseek"], ModelArchitecture.DEEPSEEK),
    ]

    for keywords, arch in patterns:
        for keyword in keywords:
            if keyword in key_lower:
                return arch

    # Try to parse config for model_type
    if config_str:
        try:
            config = json.loads(config_str)
            model_type = config.get("model_type", "").lower()
            for keywords, arch in patterns:
                for keyword in keywords:
                    if keyword in model_type:
                        return arch
        except (json.JSONDecodeError, TypeError):
            pass

    return ModelArchitecture.UNKNOWN


def fetch_ndif_status() -> Dict[str, Any]:
    """Fetch current NDIF deployment status."""
    response = requests.get(NDIF_STATUS_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def get_available_models(
    status: Optional[Dict[str, Any]] = None,
    hot_only: bool = True,
) -> List[ModelInfo]:
    """Get list of available models from NDIF status.

    Args:
        status: Pre-fetched status dict, or None to fetch fresh
        hot_only: If True, only return HOT (active) models

    Returns:
        List of ModelInfo for available models
    """
    if status is None:
        status = fetch_ndif_status()

    deployments = status.get("deployments", {})
    models = []

    for key, data in deployments.items():
        level = DeploymentLevel(data.get("deployment_level", "COLD"))

        if hot_only and level != DeploymentLevel.HOT:
            continue

        model = ModelInfo(
            model_key=data.get("repo_id", key),
            repo_id=data.get("repo_id", key),
            deployment_level=level,
            application_state=data.get("application_state", "UNKNOWN"),
            n_params=data.get("n_params"),
            dedicated=data.get("dedicated", False),
            architecture=detect_architecture(key, data.get("config")),
        )
        models.append(model)

    return models


def get_models_by_architecture(
    models: List[ModelInfo],
) -> Dict[ModelArchitecture, List[ModelInfo]]:
    """Group models by architecture."""
    by_arch: Dict[ModelArchitecture, List[ModelInfo]] = {}
    for model in models:
        if model.architecture not in by_arch:
            by_arch[model.architecture] = []
        by_arch[model.architecture].append(model)
    return by_arch


def select_test_models(
    models: List[ModelInfo],
    max_per_architecture: int = 2,
    prefer_smaller: bool = True,
) -> List[ModelInfo]:
    """Select a representative subset of models for testing.

    Args:
        models: List of available models
        max_per_architecture: Maximum models to test per architecture
        prefer_smaller: If True, prefer smaller models (faster tests)

    Returns:
        Selected models for testing
    """
    by_arch = get_models_by_architecture(models)
    selected = []

    for arch, arch_models in by_arch.items():
        # Sort by parameter count (smaller first if prefer_smaller)
        sorted_models = sorted(
            arch_models,
            key=lambda m: m.n_params or float("inf"),
            reverse=not prefer_smaller,
        )
        selected.extend(sorted_models[:max_per_architecture])

    return selected


def get_baseline_models(
    all_models: Optional[List[ModelInfo]] = None,
) -> List[ModelInfo]:
    """Get the baseline models that should always be tested.

    Args:
        all_models: List of all available models from NDIF (fetches if None)

    Returns:
        List of ModelInfo for baseline models (those that exist in NDIF)
    """
    if all_models is None:
        all_models = get_available_models(hot_only=False)

    # Create lookup by model_key
    by_key = {m.model_key: m for m in all_models}

    baseline = []
    for model_key in BASELINE_MODELS:
        if model_key in by_key:
            baseline.append(by_key[model_key])

    return baseline


def get_test_models(
    status: Optional[Dict[str, Any]] = None,
    include_extra_hot: bool = True,
    max_extra_per_architecture: int = 1,
) -> List[ModelInfo]:
    """Get models to test: baseline models + optionally extra hot models.

    Args:
        status: Pre-fetched status dict, or None to fetch fresh
        include_extra_hot: Whether to include additional hot models beyond baseline
        max_extra_per_architecture: Max extra models per architecture

    Returns:
        List of models to test
    """
    if status is None:
        status = fetch_ndif_status()

    all_models = get_available_models(status, hot_only=False)
    hot_models = get_available_models(status, hot_only=True)

    # Start with baseline models
    baseline = get_baseline_models(all_models)
    baseline_keys = {m.model_key for m in baseline}

    result = list(baseline)

    # Add extra hot models (not in baseline)
    if include_extra_hot:
        extra_hot = [m for m in hot_models if m.model_key not in baseline_keys]
        extra_selected = select_test_models(extra_hot, max_per_architecture=max_extra_per_architecture)
        result.extend(extra_selected)

    return result


def print_status_summary(models: List[ModelInfo]) -> None:
    """Print a summary of model availability."""
    by_arch = get_models_by_architecture(models)

    print(f"\nNDIF Models Available: {len(models)}")
    print("-" * 50)

    for arch, arch_models in sorted(by_arch.items(), key=lambda x: x[0].value):
        available = [m for m in arch_models if m.is_available]
        print(f"{arch.value:12} : {len(available)} available")
        for m in available[:3]:  # Show first 3
            params = f"({m.n_params / 1e9:.1f}B)" if m.n_params else ""
            print(f"             - {m.short_name} {params}")
        if len(available) > 3:
            print(f"             ... and {len(available) - 3} more")
