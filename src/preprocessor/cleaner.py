"""
Data preprocessor - cleans, deduplicates, and chunks data for LLM input.
Token counting with tiktoken to minimize API costs.
"""

import re
from dataclasses import dataclass

import tiktoken

from src.collector.base import CollectedData
from src.utils.logger import setup_logger

logger = setup_logger("preprocessor")


@dataclass
class ProcessedChunk:
    """Text chunk ready for LLM."""
    chunk_id: str
    text: str
    token_count: int
    data_type: str
    source: str
    metadata: dict


class DataPreprocessor:
    """
    Cleans and chunks collected data before sending to the LLM.
    Steps: filter -> dedup -> clean -> chunk
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.max_chunk_tokens = self.config.get("max_text_length", 3000)
        self.min_relevance = self.config.get("min_relevance_score", 0.3)

        # tiktoken으로 정확한 토큰 수 계산
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def process(self, collected_data: list[CollectedData]) -> list[ProcessedChunk]:
        """Run the full preprocessing pipeline."""
        logger.info(f"Processing {len(collected_data)} items...")

        # 1. reliability 낮은거 필터
        filtered = self._filter_by_reliability(collected_data)
        logger.info(
            f"Reliability filter: {len(filtered)} kept "
            f"({len(collected_data) - len(filtered)} removed)"
        )

        # 2. 중복 제거
        deduped = self._deduplicate(filtered)
        logger.info(
            f"Dedup: {len(deduped)} unique "
            f"({len(filtered) - len(deduped)} dupes)"
        )

        # 3. 텍스트 정리
        cleaned = [self._clean_text(d) for d in deduped]

        # 4. 토큰 제한에 맞게 분할
        chunks = self._chunk_data(cleaned)
        logger.info(
            f"Chunked: {len(chunks)} chunks, "
            f"{sum(c.token_count for c in chunks):,} total tokens"
        )

        return chunks

    def _filter_by_reliability(self, data: list[CollectedData]) -> list[CollectedData]:
        return [d for d in data if d.reliability_score >= self.min_relevance]

    def _deduplicate(self, data: list[CollectedData]) -> list[CollectedData]:
        seen: set[str] = set()
        unique: list[CollectedData] = []

        for item in data:
            fp = self._fingerprint(item.raw_text)
            if fp not in seen:
                seen.add(fp)
                unique.append(item)

        return unique

    def _fingerprint(self, text: str) -> str:
        """Normalize text for dedup comparison."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return normalized[:200]  # 처음 200자면 충분

    def _clean_text(self, data: CollectedData) -> CollectedData:
        text = data.raw_text

        # 공백 정리
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # URL, 이메일 제거
        text = re.sub(r"https?://\S+", "[URL]", text)
        text = re.sub(r"\S+@\S+", "[EMAIL]", text)

        # 너무 긴 텍스트 자르기
        tokens = self._count_tokens(text)
        if tokens > self.max_chunk_tokens * 2:
            text = self._truncate_to_tokens(text, self.max_chunk_tokens * 2)

        data.raw_text = text.strip()
        return data

    def _chunk_data(self, data: list[CollectedData]) -> list[ProcessedChunk]:
        chunks: list[ProcessedChunk] = []

        for i, item in enumerate(data):
            tc = self._count_tokens(item.raw_text)

            if tc <= self.max_chunk_tokens:
                chunks.append(ProcessedChunk(
                    chunk_id=f"{item.data_type}_{i}_0",
                    text=item.raw_text,
                    token_count=tc,
                    data_type=item.data_type,
                    source=item.source,
                    metadata=item.metadata,
                ))
            else:
                # 큰 텍스트는 문단 단위로 분할
                parts = self._split_text(item.raw_text, self.max_chunk_tokens)
                for j, part in enumerate(parts):
                    chunks.append(ProcessedChunk(
                        chunk_id=f"{item.data_type}_{i}_{j}",
                        text=part,
                        token_count=self._count_tokens(part),
                        data_type=item.data_type,
                        source=item.source,
                        metadata={**item.metadata, "chunk_index": j, "total_chunks": len(parts)},
                    ))

        return chunks

    def _split_text(self, text: str, max_tokens: int) -> list[str]:
        """Split at paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            pt = self._count_tokens(para)

            if current_tokens + pt > max_tokens and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_tokens = pt
            else:
                current.append(para)
                current_tokens += pt

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    def get_total_tokens(self, chunks: list[ProcessedChunk]) -> int:
        return sum(c.token_count for c in chunks)
