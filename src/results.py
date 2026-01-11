"""Result collection and error classification for NDIF Monitor."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import re


class Status(Enum):
    """Test result status levels."""
    OK = "OK"              # Test passed, performance normal
    SLOW = "SLOW"          # Test passed but exceeded time threshold
    DEGRADED = "DEGRADED"  # Partial failure or unexpected behavior
    FAILED = "FAILED"      # Test failed with error
    UNAVAILABLE = "UNAVAILABLE"  # Model/service not reachable
    COLD = "COLD"          # Model intentionally offline (matches NDIF status)


class ErrorCategory(Enum):
    """Classification of error types for actionable diagnostics."""
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"      # Model not in NDIF deployment
    SERIALIZATION_ERROR = "SERIALIZATION_ERROR"  # Module whitelisting issues
    TIMEOUT = "TIMEOUT"                        # Request exceeded time limit
    SHAPE_MISMATCH = "SHAPE_MISMATCH"          # Unexpected tensor shapes
    VALUE_ERROR = "VALUE_ERROR"                # Unexpected values/NaN
    CONNECTION_ERROR = "CONNECTION_ERROR"      # Network issues
    AUTH_ERROR = "AUTH_ERROR"                  # API key issues
    IMPORT_ERROR = "IMPORT_ERROR"              # Package import failures
    UNKNOWN = "UNKNOWN"                        # Uncategorized errors


@dataclass
class ScenarioResult:
    """Result of a scenario test for a model."""
    status: Status
    duration_ms: int
    last_checked: datetime
    last_success: Optional[datetime] = None
    error_category: Optional[ErrorCategory] = None
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        # Truncate details to avoid bloated JSON from full tracebacks
        details = self.details
        if details and len(details) > 500:
            details = details[:500] + "... [truncated]"
        return {
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "last_checked": self.last_checked.isoformat() + "Z",
            "last_success": self.last_success.isoformat() + "Z" if self.last_success else None,
            "error_category": self.error_category.value if self.error_category else None,
            "details": details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioResult":
        """Create from dictionary."""
        return cls(
            status=Status(data["status"]),
            duration_ms=data["duration_ms"],
            last_checked=datetime.fromisoformat(data["last_checked"].rstrip("Z")),
            last_success=datetime.fromisoformat(data["last_success"].rstrip("Z")) if data.get("last_success") else None,
            error_category=ErrorCategory(data["error_category"]) if data.get("error_category") else None,
            details=data.get("details"),
        )


@dataclass
class ModelStatus:
    """Status of a single model, tracking all scenarios."""
    model: str
    last_updated: datetime
    nnsight_version: str
    scenarios: Dict[str, ScenarioResult] = field(default_factory=dict)

    @property
    def overall_status(self) -> Status:
        """Determine overall model status from scenarios."""
        if not self.scenarios:
            return Status.UNAVAILABLE

        statuses = [s.status for s in self.scenarios.values()]

        # If any scenario is unavailable, model is unavailable
        if Status.UNAVAILABLE in statuses:
            return Status.UNAVAILABLE
        # If any scenario failed, model is failed
        if Status.FAILED in statuses:
            return Status.FAILED
        # If any scenario is degraded, model is degraded
        if Status.DEGRADED in statuses:
            return Status.DEGRADED
        # If any scenario is slow, model is slow
        if Status.SLOW in statuses:
            return Status.SLOW
        # If all scenarios are cold, model is cold
        if all(s == Status.COLD for s in statuses):
            return Status.COLD
        # All OK
        return Status.OK

    @property
    def last_all_ok(self) -> Optional[datetime]:
        """When was the model last fully healthy (all scenarios OK)."""
        if not self.scenarios:
            return None

        # Get the earliest last_success among OK scenarios
        # If all scenarios have a last_success, return the minimum
        successes = [s.last_success for s in self.scenarios.values() if s.last_success]
        if not successes or len(successes) != len(self.scenarios):
            return None
        return min(successes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "model": self.model,
            "last_updated": self.last_updated.isoformat() + "Z",
            "nnsight_version": self.nnsight_version,
            "overall_status": self.overall_status.value,
            "last_all_ok": self.last_all_ok.isoformat() + "Z" if self.last_all_ok else None,
            "scenarios": {k: v.to_dict() for k, v in self.scenarios.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelStatus":
        """Create from dictionary."""
        return cls(
            model=data["model"],
            last_updated=datetime.fromisoformat(data["last_updated"].rstrip("Z")),
            nnsight_version=data.get("nnsight_version", "unknown"),
            scenarios={k: ScenarioResult.from_dict(v) for k, v in data.get("scenarios", {}).items()},
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> Optional["ModelStatus"]:
        """Load from JSON file, returns None if file doesn't exist."""
        if not Path(path).exists():
            return None
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class TestResult:
    """Result of a single test execution."""
    model: str
    scenario: str
    status: Status
    duration_ms: int
    error_category: Optional[ErrorCategory] = None
    details: Optional[str] = None
    output: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "model": self.model,
            "scenario": self.scenario,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error_category": self.error_category.value if self.error_category else None,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() + "Z",
        }


@dataclass
class MonitorRun:
    """Results of a complete monitoring run."""
    timestamp: datetime
    nnsight_version: str
    duration_seconds: float
    tests: List[TestResult] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, int]:
        """Count of tests by status."""
        counts = {s.value.lower(): 0 for s in Status}
        for test in self.tests:
            counts[test.status.value.lower()] += 1
        counts["total"] = len(self.tests)
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp.isoformat() + "Z",
            "nnsight_version": self.nnsight_version,
            "duration_seconds": round(self.duration_seconds, 1),
            "summary": self.summary,
            "tests": [t.to_dict() for t in self.tests],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())


def classify_error(error_text: str) -> ErrorCategory:
    """Classify an error message into an ErrorCategory."""
    error_lower = error_text.lower()

    # Check for specific error patterns
    patterns = [
        (r"not whitelisted|whitelist", ErrorCategory.SERIALIZATION_ERROR),
        (r"serializ|pickle|marshal", ErrorCategory.SERIALIZATION_ERROR),
        (r"timeout|timed out|deadline exceeded", ErrorCategory.TIMEOUT),
        (r"connection|network|unreachable|refused", ErrorCategory.CONNECTION_ERROR),
        (r"auth|api.key|unauthorized|forbidden|401|403", ErrorCategory.AUTH_ERROR),
        (r"not loaded|not available|not deployed|not found.*model", ErrorCategory.MODEL_NOT_LOADED),
        (r"shape|dimension|size mismatch|expected.*got", ErrorCategory.SHAPE_MISMATCH),
        (r"nan|inf|invalid value|value error", ErrorCategory.VALUE_ERROR),
        (r"import|module|no module named|cannot import", ErrorCategory.IMPORT_ERROR),
    ]

    for pattern, category in patterns:
        if re.search(pattern, error_lower):
            return category

    return ErrorCategory.UNKNOWN


def determine_status(
    success: bool,
    duration_ms: int,
    error_text: Optional[str] = None,
) -> Status:
    """Determine test status based on success.

    Note: SLOW status is determined at analysis/dashboard time, not here.
    This allows thresholds to be adjusted without re-running tests.
    """
    if not success:
        if error_text:
            category = classify_error(error_text)
            if category == ErrorCategory.MODEL_NOT_LOADED:
                return Status.UNAVAILABLE
        return Status.FAILED

    return Status.OK


def model_to_filename(model_name: str) -> str:
    """Convert model name to safe filename."""
    # Replace / with -- and other special chars
    safe = model_name.replace("/", "--").replace(":", "_")
    return f"{safe}.json"


def filename_to_model(filename: str) -> str:
    """Convert filename back to model name."""
    name = filename.replace(".json", "")
    return name.replace("--", "/").replace("_", ":")
