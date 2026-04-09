"""
Base collector interface for all data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CollectedData:
    """Standardized container for collected data from any source."""

    source: str
    data_type: str  # "financial", "news", "industry"
    content: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    collected_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    reliability_score: float = 1.0  # 0.0 to 1.0


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.

    Each collector is responsible for gathering data from a specific
    source type and normalizing it into CollectedData format.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._cache: dict[str, CollectedData] = {}

    @abstractmethod
    async def collect(self, company: str, **kwargs) -> list[CollectedData]:
        """
        Collect data for a given company.

        Args:
            company: Company name or ticker symbol

        Returns:
            List of CollectedData objects
        """
        pass

    def _get_cache_key(self, company: str, **kwargs) -> str:
        """Generate a cache key to avoid duplicate API calls."""
        return f"{self.__class__.__name__}:{company}:{hash(str(kwargs))}"

    def _check_cache(self, key: str) -> CollectedData | None:
        """Check if data exists in cache."""
        return self._cache.get(key)

    def _set_cache(self, key: str, data: CollectedData) -> None:
        """Store data in cache."""
        self._cache[key] = data
