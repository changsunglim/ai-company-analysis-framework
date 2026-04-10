"""Tests for the preprocessor."""

import unittest
from src.collector.base import CollectedData
from src.preprocessor.cleaner import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        self.pp = DataPreprocessor(
            {"max_text_length": 500, "min_relevance_score": 0.3}
        )

    def test_filter_by_reliability(self):
        data = [
            CollectedData(
                source="test", data_type="financial",
                raw_text="High quality data", reliability_score=0.9,
            ),
            CollectedData(
                source="test", data_type="news",
                raw_text="Low quality data", reliability_score=0.1,
            ),
        ]

        result = self.pp._filter_by_reliability(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].raw_text, "High quality data")

    def test_deduplication(self):
        data = [
            CollectedData(
                source="source1", data_type="news",
                raw_text="Apple reported strong Q4 earnings today",
                reliability_score=0.8,
            ),
            CollectedData(
                source="source2", data_type="news",
                raw_text="Apple reported strong Q4 earnings today",
                reliability_score=0.7,
            ),
        ]

        result = self.pp._deduplicate(data)
        self.assertEqual(len(result), 1)

    def test_url_replacement(self):
        """URLs should be replaced with [URL]."""
        data = CollectedData(
            source="test", data_type="news",
            raw_text="Check https://example.com for details",
            reliability_score=0.8,
        )

        result = self.pp._clean_text(data)
        self.assertIn("[URL]", result.raw_text)
        self.assertNotIn("https://", result.raw_text)

    def test_full_pipeline(self):
        data = [
            CollectedData(
                source="yahoo_finance", data_type="financial",
                raw_text="Revenue: $394.3B, Net Income: $97.0B",
                reliability_score=1.0,
            ),
            CollectedData(
                source="google_news", data_type="news",
                raw_text="Apple announces new products for 2026.",
                reliability_score=0.6,
            ),
        ]

        chunks = self.pp.process(data)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(c.token_count > 0 for c in chunks))

    def test_chunk_splitting(self):
        """Long text should be split into multiple chunks."""
        long_text = "This is a test paragraph.\n\n" * 200

        data = [
            CollectedData(
                source="test", data_type="financial",
                raw_text=long_text, reliability_score=1.0,
            )
        ]

        chunks = self.pp.process(data)
        self.assertGreater(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
