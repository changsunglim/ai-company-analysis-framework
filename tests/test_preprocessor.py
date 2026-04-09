"""
Unit tests for the data preprocessor.
"""

import unittest
from src.collector.base import CollectedData
from src.preprocessor.cleaner import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Tests for DataPreprocessor functionality."""

    def setUp(self):
        self.preprocessor = DataPreprocessor(
            {"max_text_length": 500, "min_relevance_score": 0.3}
        )

    def test_filter_by_reliability(self):
        """Test that low-reliability data is filtered out."""
        data = [
            CollectedData(
                source="test",
                data_type="financial",
                raw_text="High quality data",
                reliability_score=0.9,
            ),
            CollectedData(
                source="test",
                data_type="news",
                raw_text="Low quality data",
                reliability_score=0.1,
            ),
        ]

        result = self.preprocessor._filter_by_reliability(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].raw_text, "High quality data")

    def test_deduplication(self):
        """Test that duplicate data is removed."""
        data = [
            CollectedData(
                source="source1",
                data_type="news",
                raw_text="Apple reported strong Q4 earnings today",
                reliability_score=0.8,
            ),
            CollectedData(
                source="source2",
                data_type="news",
                raw_text="Apple reported strong Q4 earnings today",
                reliability_score=0.7,
            ),
        ]

        result = self.preprocessor._deduplicate(data)
        self.assertEqual(len(result), 1)

    def test_clean_text_removes_urls(self):
        """Test that URLs are replaced with [URL] placeholder."""
        data = CollectedData(
            source="test",
            data_type="news",
            raw_text="Check https://example.com for more details",
            reliability_score=0.8,
        )

        result = self.preprocessor._clean_text(data)
        self.assertIn("[URL]", result.raw_text)
        self.assertNotIn("https://", result.raw_text)

    def test_process_full_pipeline(self):
        """Test the complete preprocessing pipeline."""
        data = [
            CollectedData(
                source="yahoo_finance",
                data_type="financial",
                raw_text="Revenue: $394.3B, Net Income: $97.0B",
                reliability_score=1.0,
            ),
            CollectedData(
                source="google_news",
                data_type="news",
                raw_text="Apple announces new product lineup for 2026.",
                reliability_score=0.6,
            ),
        ]

        chunks = self.preprocessor.process(data)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(c.token_count > 0 for c in chunks))

    def test_chunk_splitting(self):
        """Test that large texts are split into multiple chunks."""
        # Create a long text that exceeds max chunk size
        long_text = "This is a test paragraph.\n\n" * 200

        data = [
            CollectedData(
                source="test",
                data_type="financial",
                raw_text=long_text,
                reliability_score=1.0,
            )
        ]

        chunks = self.preprocessor.process(data)
        # Should produce multiple chunks
        self.assertGreater(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
