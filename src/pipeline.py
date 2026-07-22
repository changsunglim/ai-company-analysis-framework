"""
Main pipeline orchestrator.
Coordinates: collect -> preprocess -> analyze -> report
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
    4-stage company analysis pipeline.

    1. COLLECT  - financial data, news, industry context
    2. PREPROCESS - clean, dedup, chunk
    3. ANALYZE - LLM modules
    4. REPORT - markdown output
    """

    def __init__(self, config_path: str = "config/config.yaml", api_key: str | None = None):
        self.config = self._load_config(config_path)

        collector_cfg = self.config.get("collector", {})
        self.collectors = {
            "financial": FinancialCollector(collector_cfg.get("financial", {})),
            "news": NewsCollector(collector_cfg.get("news", {})),
            "industry": IndustryCollector(collector_cfg.get("industry", {})),
        }

        self.preprocessor = DataPreprocessor(self.config.get("preprocessor", {}))
        self.analyzer = LLMAnalyzer(self.config.get("analyzer", {}), api_key=api_key)
        self.reporter = ReportGenerator(self.config.get("reporter", {}))

        self.metrics: dict[str, float] = {}

    def _load_config(self, config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Config loaded: {config_path}")
            return config
        else:
            logger.warning(f"Config not found: {config_path}, using defaults")
            return {}

    async def run(
        self,
        company: str,
        ticker: str | None = None,
        modules: list[str] | None = None,
    ) -> str:
        """
        Run the full pipeline.

        Args:
            company: Company name (for news search)
            ticker: Stock ticker (for financial data). Falls back to company name.
            modules: Specific modules to run (None = all)

        Returns:
            Path to generated report
        """
        ticker = ticker or company
        pipeline_start = time.time()

        logger.info(f"{'=' * 50}")
        logger.info(f"Pipeline start: {company} ({ticker})")
        logger.info(f"{'=' * 50}")

        # stage 1
        collected = await self._stage_collect(company, ticker)
        if not collected:
            raise RuntimeError(
                f"No data for {company}. Check ticker/network."
            )

        # stage 2
        chunks = self._stage_preprocess(collected)
        if not chunks:
            raise RuntimeError("Preprocessing produced nothing usable.")

        # stage 3
        results = await self._stage_analyze(company, chunks, modules)
        if not results:
            raise RuntimeError("Analysis produced no results.")

        # stage 4
        report_path = self._stage_report(company, results)

        total_time = time.time() - pipeline_start
        self.metrics["total_pipeline_time"] = round(total_time, 2)

        logger.info(f"{'=' * 50}")
        logger.info(f"Done in {total_time:.1f}s -> {report_path}")
        logger.info(f"{'=' * 50}")

        self._print_summary()
        return report_path

    async def _stage_collect(
        self, company: str, ticker: str
    ) -> list[CollectedData]:
        """Stage 1: data collection from all sources."""
        start = time.time()
        logger.info("[1/4] Collecting data...")

        all_data: list[CollectedData] = []
        cfg = self.config.get("collector", {})

        # 동시에 수집
        tasks = []
        if cfg.get("financial", {}).get("enabled", True):
            tasks.append(self.collectors["financial"].collect(ticker))
        if cfg.get("news", {}).get("enabled", True):
            tasks.append(self.collectors["news"].collect(company))
        if cfg.get("industry", {}).get("enabled", True):
            tasks.append(self.collectors["industry"].collect(ticker))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Collector error: {r}")
            elif isinstance(r, list):
                all_data.extend(r)

        elapsed = time.time() - start
        self.metrics["collect_time"] = round(elapsed, 2)
        self.metrics["data_points"] = len(all_data)

        logger.info(
            f"[1/4] Done: {len(all_data)} items in {elapsed:.1f}s"
        )
        return all_data

    def _stage_preprocess(self, data: list[CollectedData]) -> list:
        """Stage 2: clean + chunk."""
        start = time.time()
        logger.info("[2/4] Preprocessing...")

        chunks = self.preprocessor.process(data)

        elapsed = time.time() - start
        self.metrics["preprocess_time"] = round(elapsed, 2)
        self.metrics["chunks"] = len(chunks)
        self.metrics["total_tokens_input"] = sum(c.token_count for c in chunks)

        logger.info(
            f"[2/4] Done: {len(chunks)} chunks, "
            f"{self.metrics['total_tokens_input']:,} tokens ({elapsed:.1f}s)"
        )
        return chunks

    async def _stage_analyze(
        self, company: str, chunks: list, modules: list[str] | None
    ) -> dict[str, str]:
        """Stage 3: LLM analysis."""
        start = time.time()
        logger.info("[3/4] Analyzing...")

        analysis_modules = modules or self.config.get("analyzer", {}).get("modules")

        results = await self.analyzer.analyze(
            company=company, chunks=chunks, modules=analysis_modules,
        )

        elapsed = time.time() - start
        self.metrics["analyze_time"] = round(elapsed, 2)
        self.metrics["modules_completed"] = len(results)
        self.metrics.update(self.analyzer.get_usage_stats())

        logger.info(f"[3/4] Done: {len(results)} modules ({elapsed:.1f}s)")
        return results

    def _stage_report(self, company: str, results: dict[str, str]) -> str:
        """Stage 4: generate report."""
        start = time.time()
        logger.info("[4/4] Generating report...")

        path = self.reporter.generate(
            company=company,
            analysis_results=results,
            usage_stats=self.analyzer.get_usage_stats(),
            sources=["Yahoo Finance", "Google News"],
        )

        elapsed = time.time() - start
        self.metrics["report_time"] = round(elapsed, 2)
        logger.info(f"[4/4] Report done ({elapsed:.1f}s)")
        return path

    def _print_summary(self) -> None:
        logger.info("\n--- Summary ---")
        logger.info(f"  Time: {self.metrics.get('total_pipeline_time', 0)}s")
        logger.info(f"  Data: {self.metrics.get('data_points', 0)} items")
        logger.info(f"  Chunks: {self.metrics.get('chunks', 0)}")
        logger.info(f"  Modules: {self.metrics.get('modules_completed', 0)}")
        logger.info(f"  API calls: {self.metrics.get('api_calls', 0)}")
        logger.info(f"  Tokens: {self.metrics.get('total_tokens', 'N/A')}")
        logger.info(f"  Cost: ${self.metrics.get('estimated_cost_usd', 0):.4f}")
