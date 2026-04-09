"""
Data preprocessing and cleaning pipeline.

Normalizes, deduplicates, and chunks collected data before
sending to the LLM for analysis. Designed to minimize token
usage while preserving analytical value.
"""

import re
from dataclasses import dataclass

import tiktoken

from src.collector.base import CollectedData
from src.utils.logger import setup_logger

logger = setup_logger("preprocessor")


@dataclass
class ProcessedChunk:
    """A preprocessed text chunk ready for LLM analysis."""

    chunk_id: str
    text: str
    token_count: int
    data_type: str
    source: str
    metadata: dict


class DataPreprocessor:
    """
    Preprocesses collected data for LLM consumption.

    Key responsibilities:
    - Remove duplicate information across sources
    - Clean and normalize text
    - Chunk data to fit within token limits
    - Prioritize high-reliability data
    - Estimate token counts for cost optimization
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.max_chunk_tokens = self.config.get("max_text_length", 3000)
        self.min_relevance = self.config.get("min_relevance_score", 0.3)

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def process(
        self, collected_data: list[CollectedData]
    ) -> list[ProcessedChunk]:
        """
        Process all collected data into LLM-ready chunks.

        Args:
            collected_data: Raw data from collectors

        Returns:
            List of ProcessedChunk objects optimized for LLM analysis
        """
        logger.info(
            f"Preprocessing {len(collected_data)} data items..."
        )

        # Step 1: Filter by reliability
        filtered = self._filter_by_reliability(collected_data)
        logger.info(
            f"After reliability filter: {len(filtered)} items "
            f"(removed {len(collected_data) - len(filtered)})"
        )

        # Step 2: Deduplicate
        deduped = self._deduplicate(filtered)
        logger.info(
            f"After deduplication: {len(deduped)} items "
            f"(removed {len(filtered) - len(deduped)})"
        )

        # Step 3: Clean text
        cleaned = [self._clean_text(d) for d in deduped]

        # Step 4: Chunk for token limits
        chunks = self._chunk_data(cleaned)
        logger.info(
            f"Created {len(chunks)} chunks "
            f"(total tokens: {sum(c.token_count for c in chunks):,})"
        )

        return chunks

    def _filter_by_reliability(
        self, data: list[CollectedData]
    ) -> list[CollectedData]:
        """Remove data below minimum reliability threshold."""
        return [
            d for d in data if d.reliability_score >= self.min_relevance
        ]

    def _deduplicate(
        self, data: list[CollectedData]
    ) -> list[CollectedData]:
        """Remove duplicate data based on content similarity."""
        seen_texts: set[str] = set()
        unique_data: list[CollectedData] = []

        for item in data:
            # Create a normalized fingerprint for dedup
            fingerprint = self._create_fingerprint(item.raw_text)

            if fingerprint not in seen_texts:
                seen_texts.add(fingerprint)
                unique_data.append(item)

        return unique_data

    def _create_fingerprint(self, text: str) -> str:
        """Create a normalized fingerprint for deduplication."""
        # Normalize whitespace, lowercase, remove punctuation
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        normalized = re.sub(r"[^\w\s]", "", normalized)
        # Use first 200 chars as fingerprint (good enough for dedup)
        return normalized[:200]

    def _clean_text(self, data: CollectedData) -> CollectedData:
        """Clean and normalize text content."""
        text = data.raw_text

        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove common noise patterns
        text = re.sub(r"https?://\S+", "[URL]", text)  # Replace URLs
        text = re.sub(r"\S+@\S+", "[EMAIL]", text)  # Replace emails

        # Trim very long texts
        tokens = self._count_tokens(text)
        if tokens > self.max_chunk_tokens * 2:
            text = self._truncate_to_tokens(
                text, self.max_chunk_tokens * 2
            )

        data.raw_text = text.strip()
        return data

    def _chunk_data(
        self, data: list[CollectedData]
    ) -> list[ProcessedChunk]:
        """Split data into token-limited chunks."""
        chunks: list[ProcessedChunk] = []

        for i, item in enumerate(data):
            token_count = self._count_tokens(item.raw_text)

            if token_count <= self.max_chunk_tokens:
                # Fits in a single chunk
                chunks.append(
                    ProcessedChunk(
                        chunk_id=f"{item.data_type}_{i}_0",
                        text=item.raw_text,
                        token_count=token_count,
                        data_type=item.data_type,
                        source=item.source,
                        metadata=item.metadata,
                    )
                )
            else:
                # Split into multiple chunks
                sub_chunks = self._split_text(
                    item.raw_text, self.max_chunk_tokens
                )
                for j, sub_text in enumerate(sub_chunks):
                    chunks.append(
                        ProcessedChunk(
                            chunk_id=f"{item.data_type}_{i}_{j}",
                            text=sub_text,
                            token_count=self._count_tokens(sub_text),
                            data_type=item.data_type,
                            source=item.source,
                            metadata={
                                **item.metadata,
                                "chunk_index": j,
                                "total_chunks": len(sub_chunks),
                            },
                        )
                    )

        return chunks

    def _split_text(
        self, text: str, max_tokens: int
    ) -> list[str]:
        """Split text into chunks at paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if current_tokens + para_tokens > max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.tokenizer.encode(text))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to a maximum number of tokens."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    def get_total_tokens(self, chunks: list[ProcessedChunk]) -> int:
        """Calculate total token count across all chunks."""
        return sum(c.token_count for c in chunks)
