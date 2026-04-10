"""
Industry/competitive landscape collector.
Compares target company with sector competitors using yfinance.
"""

import asyncio
from typing import Any

import yfinance as yf

from src.collector.base import BaseCollector, CollectedData
from src.utils.logger import setup_logger

logger = setup_logger("industry_collector")


# 섹터별 주요 경쟁사 매핑 (하드코딩이긴 한데 일단 이렇게...)
# TODO: 나중에 동적으로 가져오는 방법 찾기
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
    """Collects competitor data for comparison."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.max_competitors = self.config.get("max_competitors", 5)

    async def collect(self, company: str, **kwargs) -> list[CollectedData]:
        logger.info(f"Collecting industry data: {company}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._gather_industry_data, company
        )

        if not result:
            return []
        return [result]

    def _gather_industry_data(self, ticker: str) -> CollectedData | None:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")

            competitors = self._find_competitors(ticker, sector, info)
            comparison = self._build_comparison(ticker, info, competitors)

            text = (
                f"=== Industry: {ticker} ===\n\n"
                f"Sector: {sector}\n"
                f"Industry: {industry}\n\n"
                f"--- Competitors ---\n"
            )

            for c in comparison:
                text += (
                    f"\n{c['ticker']} ({c.get('name', 'N/A')}):\n"
                    f"  Market Cap: {self._fmt_number(c.get('market_cap'))}\n"
                    f"  P/E: {c.get('pe_ratio', 'N/A')}\n"
                    f"  Revenue Growth: {c.get('revenue_growth', 'N/A')}\n"
                    f"  Profit Margin: {c.get('profit_margin', 'N/A')}\n"
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
                metadata={"ticker": ticker, "competitor_count": len(comparison)},
            )

        except Exception as e:
            logger.error(f"Industry data failed: {e}")
            return None

    def _find_competitors(self, ticker: str, sector: str, info: dict) -> list[str]:
        comps = SECTOR_COMPETITORS.get(sector, [])
        comps = [c for c in comps if c != ticker]
        return comps[:self.max_competitors]

    def _build_comparison(
        self, ticker: str, info: dict, competitors: list[str]
    ) -> list[dict[str, Any]]:
        results = []

        # 타겟 회사 먼저
        results.append(self._get_metrics(ticker, info))

        for comp_ticker in competitors:
            try:
                comp = yf.Ticker(comp_ticker)
                comp_info = comp.info
                if comp_info:
                    results.append(self._get_metrics(comp_ticker, comp_info))
            except Exception as e:
                logger.debug(f"Competitor {comp_ticker} failed: {e}")

        return results

    def _get_metrics(self, ticker: str, info: dict) -> dict[str, Any]:
        """Extract comparison metrics for one company."""
        return {
            "ticker": ticker,
            "name": info.get("longName", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": (
                round(info["trailingPE"], 2) if info.get("trailingPE") else "N/A"
            ),
            "revenue_growth": (
                f"{info['revenueGrowth']:.1%}" if info.get("revenueGrowth") else "N/A"
            ),
            "profit_margin": (
                f"{info['profitMargins']:.1%}" if info.get("profitMargins") else "N/A"
            ),
            "operating_margin": (
                f"{info['operatingMargins']:.1%}" if info.get("operatingMargins") else "N/A"
            ),
            "roe": (
                f"{info['returnOnEquity']:.1%}" if info.get("returnOnEquity") else "N/A"
            ),
        }

    @staticmethod
    def _fmt_number(n: int | float | None) -> str:
        if n is None:
            return "N/A"
        if n >= 1e12:
            return f"${n/1e12:.1f}T"
        if n >= 1e9:
            return f"${n/1e9:.1f}B"
        if n >= 1e6:
            return f"${n/1e6:.1f}M"
        return f"${n:,.0f}"
