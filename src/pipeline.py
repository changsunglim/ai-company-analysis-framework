"""
Main analysis pipeline orchestrator.

Coordinates the full data collection → preprocessing → analysis → report
pipeline with configurable stages and comprehensive error handling.
"""

import asyncio
import time
from pathlib import Path

import yaml

from src.collector import FinancialCollector, NewsCollector, IndustryCollector
from src.collector.base import CollectedData
from src.preprocessor import DataPreprocessor
from src.analyzer import LLMAnalyzer
from src.reporter import ReportGenerator
from src.utils.logger import setup_logger

logger = setup_logger("pipeline")


class AnalysisPipeline:
    """
    End-to-end company analysis pipeline.

    Stages:
    1. COLLECT  — Gather financial data, news, and industry context
    2. PREPROCESS — Clean, deduplicate, and chunk data
    3. ANALYZE — Run LLM-powered analysis modules
    4. REPORT — Generate formatted output report

    Each stage is independently configurable and the pipeline
    tracks performance metrics throughout execution.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)

        # Initialize pipeline components
        collector_config = self.config.get("collector", {})
        self.collectors = {
            "financial": FinancialCollector(
                collector_config.get("financial", {})
            ),
            "news": NewsCollector(collector_config.get("news", {})),
            "industry": IndustryCollector(
                collector_config.get("industry", {})
            ),
        }

        self.preprocessor = DataPreprocessor(
            self.config.get("preprocessor", {})
        )
        self.analyzer = LLMAnalyzer(self.config.get("analyzer", {}))
        self.reporter = ReportGenerator(
            self.config.get("reporter", {})
        )

        # Pipeline metrics
        self.metrics: dict[str, float] = {}

    def _load_config(self, config_path: str) -> dict:
        """Load pipeline configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        else:
            logger.warning(
                f"Config file not found: {config_path}. "
                "Using defaults."
            )
            return {}

    async def run(
        self,
        company: str,
        ticker: str | None = None,
        modules: list[str] | None = None,
    ) -> str:
        """
        Execute the full analysis pipeline.

        Args:
            company: Company name (used for news search)
            ticker: Stock ticker symbol (used for financial data).
                    If not provided, company name is used as ticker.
            modules: Specific analysis modules to run

        Returns:
            Path to the generated report
        """
        ticker = ticker or company
        pipeline_start = time.time()

        logger.info(f"{'=' * 60}")
        logger.info(f"Starting analysis pipeline for: {company} ({ticker})")
        logger.info(f"{'=' * 60}")

        # Stage 1: Collect
        collected_data = await self._stage_collect(company, ticker)
        if not collected_data:
            raise RuntimeError(
                f"No data collected for {company}. "
                "Check the ticker symbol and network connection."
            )

        # Stage 2: Preprocess
        chunks = self._stage_preprocess(collected_data)
        if not chunks:
            raise RuntimeError("Preprocessing produced no usable chunks.")

        # Stage 3: Analyze
        analysis_results = await self._stage_analyze(
            company, chunks, modules
        )
        if not analysis_results:
            raise RuntimeError("Analysis produced no results.")

        # Stage 4: Report
        report_path = self._stage_report(company, analysis_results)

        # Final metrics
        total_time = time.time() - pipeline_start
        self.metrics["total_pipeline_time"] = round(total_time, 2)

        logger.info(f"{'=' * 60}")
        logger.info(f"Pipeline complete in {total_time:.1f}s")
        logger.info(f"Report: {report_path}")
        logger.info(f"{'=' * 60}")

        self._print_summary()

        return report_path

    async def _stage_collect(
        self, company: str, ticker: str
    ) -> list[CollectedData]:
        """Stage 1: Collect data from all sources concurrently."""
        stage_start = time.time()
        logger.info("[1/4] Collecting data...")

        all_data: list[CollectedData] = []
        collector_config = self.config.get("collector", {})

        # Run collectors concurrently
        tasks = []

        if collector_config.get("financial", {}).get("enabled", True):
            tasks.append(self.collectors["financial"].collect(ticker))

        if collector_config.get("news", {}).get("enabled", True):
            tasks.append(self.collectors["news"].collect(company))

        if collector_config.get("industry", {}).get("enabled", True):
            tasks.append(self.collectors["industry"].collect(ticker))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Collector error: {result}")
            elif isinstance(result, list):
                all_data.extend(result)

        elapsed = time.time() - stage_start
        self.metrics["collect_time"] = round(elapsed, 2)
        self.metrics["data_points"] = len(all_data)

        logger.info(
            f"[1/4] Collection complete: {len(all_data)} data points "
            f"in {elapsed:.1f}s"
        )
        return all_data

    def _stage_preprocess(
        self, collected_data: list[CollectedData]
    ) -> list:
        """Stage 2: Preprocess and chunk data."""
        stage_start = time.time()
        logger.info("[2/4] Preprocessing data...")

        chunks = self.preprocessor.process(collected_data)

        elapsed = time.time() - stage_start
        self.metrics["preprocess_time"] = round(elapsed, 2)
        self.metrics["chunks"] = len(chunks)
        self.metrics["total_tokens_input"] = sum(
            c.token_count for c in chunks
        )

        logger.info(
            f"[2/4] Preprocessing complete: {len(chunks)} chunks, "
            f"{self.metrics['total_tokens_input']:,} tokens "
            f"in {elapsed:.1f}s"
        )
        return chunks

    async def _stage_analyze(
        self, company: str, chunks: list, modules: list[str] | None
    ) -> dict[str, str]:
        """Stage 3: Run LLM analysis."""
        stage_start = time.time()
        logger.info("[3/4] Running LLM analysis...")

        analysis_modules = modules or self.config.get(
            "analyzer", {}
        ).get("modules")

        results = await self.analyzer.analyze(
            company=company,
            chunks=chunks,
            modules=analysis_modules,
        )

        elapsed = time.time() - stage_start
        self.metrics["analyze_time"] = round(elapsed, 2)
        self.metrics["modules_completed"] = len(results)
        self.metrics.update(self.analyzer.get_usage_stats())

        logger.info(
            f"[3/4] Analysis complete: {len(results)} modules "
            f"in {elapsed:.1f}s"
        )
        return results

    def _stage_report(
        self, company: str, analysis_results: dict[str, str]
    ) -> str:
        """Stage 4: Generate report."""
        stage_start = time.time()
        logger.info("[4/4] Generating report...")

        report_path = self.reporter.generate(
            company=company,
            analysis_results=analysis_results,
            usage_stats=self.analyzer.get_usage_stats(),
            sources=["Yahoo Finance", "Google News"],
        )

        elapsed = time.time() - stage_start
        self.metrics["report_time"] = round(elapsed, 2)

        logger.info(f"[4/4] Report generated in {elapsed:.1f}s")
        return report_path

    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        logger.info("\n--- Pipeline Summary ---")
        logger.info(
            f"  Total time: {self.metrics.get('total_pipeline_time', 0)}s"
        )
        logger.info(
            f"  Data points collected: {self.metrics.get('data_points', 0)}"
        )
        logger.info(
            f"  Chunks processed: {self.metrics.get('chunks', 0)}"
        )
        logger.info(
            f"  Analysis modules: {self.metrics.get('modules_completed', 0)}"
        )
        logger.info(
            f"  API calls: {self.metrics.get('api_calls', 0)}"
        )
        logger.info(
            f"  Total tokens: {self.metrics.get('total_tokens', 'N/A')}"
        )
        logger.info(
            f"  Estimated cost: ${self.metrics.get('estimated_cost_usd', 0):.4f}"
        )
