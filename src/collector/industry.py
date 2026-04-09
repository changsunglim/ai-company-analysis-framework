"""
Industry and competitive landscape data collector.

Gathers competitor information and sector context using yfinance.
"""

import asyncio
from typing import Any

import yfinance as yf

from src.collector.base import BaseCollector, CollectedData
from src.utils.logger import setup_logger

logger = setup_logger("industry_collector")


# Common sector-to-competitors mapping for quick lookup
SECTOR_COMPETITORS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "Financial Services": ["JPM", "BAC", "GS", "MS", "C"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Industrials": ["HON", "UPS", "CAT", "GE", "BA"],
}


class IndustryCollector(BaseCollector):
    """
    Collects industry context and competitive landscape data.

    Identifies competitors in the same sector and gathers
    comparative metrics for competitive positioning analysis.
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.max_competitors = self.config.get("max_competitors", 5)

    async def collect(
        self, company: str, **kwargs
    ) -> list[CollectedData]:
        """
        Collect industry and competitor data.

        Args:
            company: Stock ticker symbol

        Returns:
            List of CollectedData with industry analysis
        """
        logger.info(f"Collecting industry data for: {company}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._gather_industry_data, company
        )

        if not result:
            return []

        return [result]

    def _gather_industry_data(self, ticker: str) -> CollectedData | None:
        """Gather industry and competitor information."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")

            # Find competitors
            competitors = self._find_competitors(ticker, sector, info)

            # Build comparison data
            comparison = self._build_comparison(ticker, info, competitors)

            text = (
                f"=== Industry Analysis for {ticker} ===\n\n"
                f"Sector: {sector}\n"
                f"Industry: {industry}\n\n"
                f"--- Competitive Landscape ---\n"
            )

            for comp in comparison:
                text += (
                    f"\n{comp['ticker']} ({comp.get('name', 'N/A')}):\n"
                    f"  Market Cap: {self._format_number(comp.get('market_cap'))}\n"
                    f"  P/E Ratio: {comp.get('pe_ratio', 'N/A')}\n"
                    f"  Revenue Growth: {comp.get('revenue_growth', 'N/A')}\n"
                    f"  Profit Margin: {comp.get('profit_margin', 'N/A')}\n"
                )

            return CollectedData(
                source="yahoo_finance",
                data_type="industry",
                content={
                    "sector": sector,
                    "industry": industry,
                    "competitors": comparison,
                },
                raw_text=text,
                metadata={
                    "ticker": ticker,
                    "competitor_count": len(comparison),
                },
            )

        except Exception as e:
            logger.error(f"Failed to gather industry data: {e}")
            return None

    def _find_competitors(
        self, ticker: str, sector: str, info: dict
    ) -> list[str]:
        """Find competitor tickers in the same sector."""
        competitors = SECTOR_COMPETITORS.get(sector, [])
        # Remove the target company from competitors
        competitors = [c for c in competitors if c != ticker]
        return competitors[: self.max_competitors]

    def _build_comparison(
        self, ticker: str, info: dict, competitors: list[str]
    ) -> list[dict[str, Any]]:
        """Build a comparison table of the company vs competitors."""
        results = []

        # Add target company first
        results.append(self._extract_comparison_metrics(ticker, info))

        # Add competitors
        for comp_ticker in competitors:
            try:
                comp = yf.Ticker(comp_ticker)
                comp_info = comp.info
                if comp_info:
                    results.append(
                        self._extract_comparison_metrics(
                            comp_ticker, comp_info
                        )
                    )
            except Exception as e:
                logger.debug(
                    f"Failed to get competitor data for {comp_ticker}: {e}"
                )

        return results

    def _extract_comparison_metrics(
        self, ticker: str, info: dict
    ) -> dict[str, Any]:
        """Extract standardized comparison metrics."""
        return {
            "ticker": ticker,
            "name": info.get("longName", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": (
                round(info["trailingPE"], 2)
                if info.get("trailingPE")
                else "N/A"
            ),
            "revenue_growth": (
                f"{info['revenueGrowth']:.1%}"
                if info.get("revenueGrowth")
                else "N/A"
            ),
            "profit_margin": (
                f"{info['profitMargins']:.1%}"
                if info.get("profitMargins")
                else "N/A"
            ),
            "operating_margin": (
                f"{info['operatingMargins']:.1%}"
                if info.get("operatingMargins")
                else "N/A"
            ),
            "roe": (
                f"{info['returnOnEquity']:.1%}"
                if info.get("returnOnEquity")
                else "N/A"
            ),
        }

    @staticmethod
    def _format_number(n: int | float | None) -> str:
        """Format large numbers with B/M/K suffixes."""
        if n is None:
            return "N/A"
        if n >= 1e12:
            return f"${n / 1e12:.1f}T"
        if n >= 1e9:
            return f"${n / 1e9:.1f}B"
        if n >= 1e6:
            return f"${n / 1e6:.1f}M"
        return f"${n:,.0f}"
