"""Base collector interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CollectedData:
    """Container for data from any source."""
    source: str
    data_type: str  # financial, news, industry
    content: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    collected_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    reliability_score: float = 1.0


class BaseCollector(ABC):
    """Base class for data collectors."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._cache: dict[str, CollectedData] = {}

    @abstractmethod
    async def collect(self, company: str, **kwargs) -> list[CollectedData]:
        """Collect data for a company. Returns list of CollectedData."""
        pass

    def _get_cache_key(self, company: str, **kwargs) -> str:
        return f"{self.__class__.__name__}:{company}:{hash(str(kwargs))}"

    def _check_cache(self, key: str) -> CollectedData | None:
        return self._cache.get(key)

    def _set_cache(self, key: str, data: CollectedData) -> None:
        self._cache[key] = data
