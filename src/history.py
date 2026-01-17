"""Historical data storage for NDIF Monitor dashboard."""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import socket
import getpass

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python < 3.9

from .results import Status, ErrorCategory


# Eastern timezone for date grouping
EASTERN = ZoneInfo("America/New_York")


def utc_to_eastern_date(timestamp: str) -> str:
    """Convert UTC timestamp to Eastern date string (YYYY-MM-DD).

    Args:
        timestamp: ISO format timestamp ending in 'Z' (UTC)

    Returns:
        Date string in YYYY-MM-DD format in Eastern timezone
    """
    # Parse UTC timestamp
    ts = timestamp.rstrip('Z')
    try:
        dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except ValueError:
        # Fallback: just use the date portion
        return timestamp[:10]

    # Convert to Eastern
    eastern_dt = dt.astimezone(EASTERN)
    return eastern_dt.strftime("%Y-%m-%d")


def utc_to_eastern_hour(timestamp: str) -> str:
    """Convert UTC timestamp to Eastern date + hour.

    Args:
        timestamp: ISO format timestamp ending in 'Z' (UTC)

    Returns:
        String in format "YYYY-MM-DD-HH" where HH is hour 0-23
    """
    ts = timestamp.rstrip('Z')
    try:
        dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except ValueError:
        return timestamp[:10] + "-0"

    eastern_dt = dt.astimezone(EASTERN)
    return eastern_dt.strftime("%Y-%m-%d") + f"-{eastern_dt.hour}"


def get_hostname() -> str:
    """Get short hostname of current machine."""
    return socket.gethostname().split('.')[0]


def get_username() -> str:
    """Get current username."""
    return getpass.getuser()


@dataclass
class HistoryEntry:
    """Single historical test result entry."""
    timestamp: str  # ISO format
    model: str
    scenario: str
    status: str  # Status enum value
    duration_ms: int
    error_category: Optional[str] = None  # ErrorCategory enum value
    details: Optional[str] = None
    host: Optional[str] = None  # Machine hostname
    user: Optional[str] = None  # Username

    def to_json_line(self) -> str:
        """Convert to JSON line (no newline)."""
        data = {
            "ts": self.timestamp,
            "m": self.model,
            "s": self.scenario,
            "st": self.status,
            "d": self.duration_ms,
        }
        if self.error_category:
            data["ec"] = self.error_category
        if self.details:
            # Truncate details for history storage
            details = self.details[:2048] if len(self.details) > 2048 else self.details
            data["det"] = details
        if self.host:
            data["h"] = self.host
        if self.user:
            data["u"] = self.user
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> "HistoryEntry":
        """Parse from JSON line."""
        data = json.loads(line)
        return cls(
            timestamp=data["ts"],
            model=data["m"],
            scenario=data["s"],
            status=data["st"],
            duration_ms=data["d"],
            error_category=data.get("ec"),
            details=data.get("det"),
            host=data.get("h"),
            user=data.get("u"),
        )


class HistoryStore:
    """Append-only history storage using JSONL format."""

    def __init__(self, history_file: Path):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: HistoryEntry) -> None:
        """Append a single entry to history."""
        with open(self.history_file, "a") as f:
            f.write(entry.to_json_line() + "\n")

    def append_many(self, entries: List[HistoryEntry]) -> None:
        """Append multiple entries to history."""
        with open(self.history_file, "a") as f:
            for entry in entries:
                f.write(entry.to_json_line() + "\n")

    def load(
        self,
        days: int = 365,
        model: Optional[str] = None,
        scenario: Optional[str] = None,
    ) -> List[HistoryEntry]:
        """Load history entries, optionally filtered.

        Args:
            days: Only load entries from last N days
            model: Filter by model name
            scenario: Filter by scenario name

        Returns:
            List of HistoryEntry objects
        """
        if not self.history_file.exists():
            return []

        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        entries = []
        with open(self.history_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = HistoryEntry.from_json_line(line)
                    # Filter by date
                    if entry.timestamp < cutoff_str:
                        continue
                    # Filter by model
                    if model and entry.model != model:
                        continue
                    # Filter by scenario
                    if scenario and entry.scenario != scenario:
                        continue
                    entries.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue  # Skip malformed lines

        return entries

    def prune(self, keep_days: int = 400) -> int:
        """Remove entries older than keep_days. Returns count removed."""
        if not self.history_file.exists():
            return 0

        cutoff = datetime.utcnow() - timedelta(days=keep_days)
        cutoff_str = cutoff.isoformat()

        # Read all entries
        kept = []
        removed = 0
        with open(self.history_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = HistoryEntry.from_json_line(line)
                    if entry.timestamp >= cutoff_str:
                        kept.append(line)
                    else:
                        removed += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        # Rewrite file with kept entries
        if removed > 0:
            with open(self.history_file, "w") as f:
                for line in kept:
                    f.write(line + "\n")

        return removed

    def get_daily_summary(
        self,
        days: int = 365,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get daily status summary for dashboard.

        Returns:
            Dict mapping date -> model -> {status, scenarios}
        """
        entries = self.load(days=days)

        # Group by date and model
        daily: Dict[str, Dict[str, Dict[str, str]]] = {}

        for entry in entries:
            # Extract date and hour from timestamp, converting to Eastern timezone
            hour_key = utc_to_eastern_hour(entry.timestamp)
            date = hour_key[:10]  # YYYY-MM-DD portion

            if date not in daily:
                daily[date] = {}
            if entry.model not in daily[date]:
                daily[date][entry.model] = {"scenarios": {}, "hours": {}}

            daily[date][entry.model]["scenarios"][entry.scenario] = entry.status

            # Track hourly status (0-23)
            hour = hour_key.split("-")[-1]  # "0" through "23"
            if hour not in daily[date][entry.model]["hours"]:
                daily[date][entry.model]["hours"][hour] = []
            daily[date][entry.model]["hours"][hour].append(entry.status)

        # Helper to compute worst status from a list of statuses
        def worst_status(statuses: List[str]) -> str:
            if "UNAVAILABLE" in statuses:
                return "UNAVAILABLE"
            if "FAILED" in statuses:
                return "FAILED"
            if "DEGRADED" in statuses:
                return "DEGRADED"
            if "SLOW" in statuses:
                return "SLOW"
            if statuses and all(s == "COLD" for s in statuses):
                return "COLD"
            return "OK" if statuses else None

        # Convert to summary format
        summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for date, models in daily.items():
            summary[date] = {}
            for model, model_data in models.items():
                scenario_statuses = list(model_data["scenarios"].values())
                overall = worst_status(scenario_statuses)

                # Compute per-hour status (0-23)
                hours_summary = {}
                for hour_num, hour_statuses in model_data["hours"].items():
                    hours_summary[hour_num] = worst_status(hour_statuses)

                summary[date][model] = {
                    "status": overall,
                    "scenarios": model_data["scenarios"],
                    "hours": hours_summary,
                }

        return summary

    def get_recent_failures(
        self,
        days: int = 7,
        limit: int = 20,
    ) -> List[HistoryEntry]:
        """Get recent failure entries for debugging."""
        entries = self.load(days=days)
        failures = [
            e for e in entries
            if e.status in ("FAILED", "UNAVAILABLE", "DEGRADED")
        ]
        # Sort by timestamp descending, take most recent
        failures.sort(key=lambda e: e.timestamp, reverse=True)
        return failures[:limit]


def estimate_storage(days: int = 365, models: int = 9, scenarios: int = 3) -> str:
    """Estimate storage requirements for history.

    At 30-min intervals:
    - entries_per_day = 48 * models * scenarios
    - entry_size ~= 100 bytes (compressed JSON)
    """
    entries_per_day = 48 * models * scenarios
    total_entries = entries_per_day * days
    bytes_per_entry = 100  # approximate
    total_bytes = total_entries * bytes_per_entry

    mb = total_bytes / (1024 * 1024)
    return f"{total_entries:,} entries, ~{mb:.1f} MB"
