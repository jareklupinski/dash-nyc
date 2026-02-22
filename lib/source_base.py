"""
Base class for NYC Map data sources.

To add a new data source:
  1. Create a new .py file in your app's sources/ directory
  2. Subclass DataSource
  3. Implement fetch() to return a list of Venue dicts
  4. The build script auto-discovers all DataSource subclasses

Each venue dict should have at minimum:
  - name: str
  - lat: float
  - lng: float
  - source: str  (identifier for this data source)

Optional but encouraged:
  - address: str
  - cuisine: str
  - borough: str
  - phone: str
  - grade: str
  - tags: list[str]  (e.g. ["bar", "restaurant"])
  - meta: dict       (anything source-specific)
"""

from __future__ import annotations

import abc
import importlib
import importlib.util
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Venue:
    """A single venue on the map."""

    name: str
    lat: float
    lng: float
    source: str
    address: str = ""
    cuisine: str = ""
    borough: str = ""
    phone: str = ""
    grade: str = ""
    zipcode: str = ""
    opened: str = ""  # ISO date string (YYYY-MM-DD) — earliest known date
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "source": self.source,
        }
        # Only include non-empty optional fields to keep JSON compact
        if self.address:
            d["address"] = self.address
        if self.cuisine:
            d["cuisine"] = self.cuisine
        if self.borough:
            d["borough"] = self.borough
        if self.phone:
            d["phone"] = self.phone
        if self.grade:
            d["grade"] = self.grade
        if self.zipcode:
            d["zipcode"] = self.zipcode
        if self.opened:
            d["opened"] = self.opened
        if self.tags:
            d["tags"] = self.tags
        if self.meta:
            d["meta"] = self.meta
        return d


class DataSource(abc.ABC):
    """Abstract base class for a data source."""

    _cache_dir: Path | None = None

    @property
    def cache_dir(self) -> Path:
        """Directory for source-internal caches (SNAP zip, geocode, etc.).

        Set by the build system before calling fetch().
        Falls back to <project_root>/.cache if not explicitly set.
        """
        if self._cache_dir is not None:
            return self._cache_dir
        return Path(__file__).resolve().parent.parent / ".cache"

    @cache_dir.setter
    def cache_dir(self, path: Path) -> None:
        self._cache_dir = path

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for this source, e.g. 'dohmh', 'sla'."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description shown in the UI."""
        ...

    @property
    def url(self) -> str:
        """Public URL for this data source (for attribution / info display).

        Override in subclasses to point to the dataset landing page.
        Returns empty string by default.
        """
        return ""

    @abc.abstractmethod
    def fetch(self) -> list[Venue]:
        """Fetch and return all venues from this source.

        Should handle its own pagination, retries, etc.
        Return as many venues as possible — no deduplication needed here.
        """
        ...


def discover_sources(sources_dir: Path | str | None = None) -> list[DataSource]:
    """Auto-discover all DataSource subclasses in a sources/ directory.

    Args:
        sources_dir: Path to the sources directory to scan.
                     If None, looks for sources/ relative to caller.
    """
    if sources_dir is None:
        sources_dir = Path(__file__).parent.parent / "sources"
    sources_dir = Path(sources_dir)

    if not sources_dir.is_dir():
        log.warning("Sources directory not found: %s", sources_dir)
        return []

    files = sorted(
        f for f in sources_dir.glob("*.py") if not f.name.startswith("_")
    )
    return discover_sources_from_files(files)


def discover_sources_from_files(files: list[Path]) -> list[DataSource]:
    """Load specific source files and instantiate their DataSource subclasses.

    Args:
        files: List of .py file paths containing DataSource subclasses.

    Returns:
        A list of instantiated DataSource objects (one per file/class).
    """
    import sys

    # Collect the module names we expect sources from
    expected_modules: set[str] = set()

    for py_file in files:
        py_file = Path(py_file)
        mod_name = f"sources.{py_file.stem}"
        expected_modules.add(mod_name)
        # If already imported, skip re-import
        if mod_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                log.exception("Failed to load source module %s", py_file.name)

    # Instantiate all DataSource subclasses whose module is in our expected set
    sources = []
    for cls in DataSource.__subclasses__():
        cls_mod = getattr(cls, "__module__", "")
        if cls_mod in expected_modules:
            try:
                sources.append(cls())
                log.info("Discovered source: %s", cls.__name__)
            except Exception:
                log.exception("Failed to instantiate source %s", cls.__name__)
    return sources
