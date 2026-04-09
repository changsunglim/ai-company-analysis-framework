"""
Financial data collector using yfinance.

Gathers key financial metrics, historical prices, and fundamental
data for company analysis.
"""

import asyncio
from typing import Any

import yfinance as yf
import pandas as pd

from src.collector.base import BaseCollector, CollectedData
from src.utils.logger import setup_logger

logger = setup_logger("financial_collector")


class FinancialCollector(BaseCollector):
    """
    Collects financial data from Yahoo Finance.

    Retrieves:
    - Company profile & description
    - Key financial metrics (P/E, market cap, margins, etc.)
    - Historical price data
    - Income statement & balance sheet summaries
    - Analyst recommendations
    """

    DEFAULT_METRICS = [
        "revenue",
        "net_income",
        "operating_margin",
        "debt_to_equity",
        "free_cash_flow",
        "pe_ratio",
        "market_cap",
    ]

    async def collect(
        self, company: str, **kwargs
    ) -> list[CollectedData]:
        """
        Collect financial data for a company ticker.

        Args:
            company: Stock ticker symbol (e.g., "AAPL", "005930.KS")

        Returns:
            List of CollectedData containing financial information
        """
        logger.info(f"Collecting financial data for: {company}")

        # Run yfinance in executor (it's synchronous)
        loop = asyncio.get_event_loop()
        ticker_data = await loop.run_in_executor(
            None, self._fetch_ticker_data, company
        )

        if not ticker_data:
            logger.warning(f"No financial data found for {company}")
            return []

        results = []

        # Company profile
        profile = self._extract_profile(ticker_data)
        if profile:
            results.append(profile)

        # Financial metrics
        metrics = self._extract_metrics(ticker_data, company)
        if metrics:
            results.append(metrics)

        # Historical prices
        history = self._extract_price_history(ticker_data, company)
        if history:
            results.append(history)

        # Analyst recommendations
        recommendations = self._extract_recommendations(
            ticker_data, company
        )
        if recommendations:
            results.append(recommendations)

        logger.info(
            f"Collected {len(results)} financial data points for {company}"
        )
        return results

    def _fetch_ticker_data(self, ticker: str) -> dict[str, Any] | None:
        """Fetch all ticker data from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or info.get("trailingPegRatio") is None and not info.get("longName"):
                # Try to validate we got real data
                pass

            data = {
                "info": info,
                "stock": stock,
            }

            # Fetch financials safely
            try:
                data["income_stmt"] = stock.income_stmt
            except Exception:
                data["income_stmt"] = None

            try:
                data["balance_sheet"] = stock.balance_sheet
            except Exception:
                data["balance_sheet"] = None

            try:
                hist = stock.history(period="1y")
                data["history"] = hist
            except Exception:
                data["history"] = None

            try:
                data["recommendations"] = stock.recommendations
            except Exception:
                data["recommendations"] = None

            return data

        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return None

    def _extract_profile(
        self, ticker_data: dict
    ) -> CollectedData | None:
        """Extract company profile information."""
        info = ticker_data.get("info", {})
        if not info:
            return None

        profile_text = (
            f"Company: {info.get('longName', 'N/A')}\n"
            f"Sector: {info.get('sector', 'N/A')}\n"
            f"Industry: {info.get('industry', 'N/A')}\n"
            f"Country: {info.get('country', 'N/A')}\n"
            f"Employees: {info.get('fullTimeEmployees', 'N/A')}\n"
            f"Website: {info.get('website', 'N/A')}\n\n"
            f"Description:\n{info.get('longBusinessSummary', 'N/A')}"
        )

        return CollectedData(
            source="yahoo_finance",
            data_type="financial",
            content={
                "name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "employees": info.get("fullTimeEmployees"),
                "website": info.get("website"),
                "description": info.get("longBusinessSummary"),
            },
            raw_text=profile_text,
            metadata={"sub_type": "profile"},
        )

    def _extract_metrics(
        self, ticker_data: dict, company: str
    ) -> CollectedData | None:
        """Extract key financial metrics."""
        info = ticker_data.get("info", {})
        if not info:
            return None

        metrics = {
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "revenue": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "free_cash_flow": info.get("freeCashflow"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }

        # Build readable text
        lines = [f"=== Financial Metrics for {company} ===\n"]
        for key, value in metrics.items():
            if value is not None:
                display_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    if "margin" in key or "growth" in key or "return" in key or "yield" in key:
                        lines.append(f"{display_key}: {value:.2%}")
                    else:
                        lines.append(f"{display_key}: {value:,.2f}")
                elif isinstance(value, int):
                    lines.append(f"{display_key}: {value:,}")
                else:
                    lines.append(f"{display_key}: {value}")

        return CollectedData(
            source="yahoo_finance",
            data_type="financial",
            content=metrics,
            raw_text="\n".join(lines),
            metadata={"sub_type": "metrics", "ticker": company},
        )

    def _extract_price_history(
        self, ticker_data: dict, company: str
    ) -> CollectedData | None:
        """Extract price history summary."""
        history = ticker_data.get("history")
        if history is None or history.empty:
            return None

        try:
            current = history["Close"].iloc[-1]
            month_ago = history["Close"].iloc[-22] if len(history) > 22 else history["Close"].iloc[0]
            year_start = history["Close"].iloc[0]

            price_summary = {
                "current_price": round(current, 2),
                "1m_change_pct": round((current - month_ago) / month_ago * 100, 2),
                "period_change_pct": round((current - year_start) / year_start * 100, 2),
                "period_high": round(history["Close"].max(), 2),
                "period_low": round(history["Close"].min(), 2),
                "avg_volume": int(history["Volume"].mean()),
            }

            text = (
                f"=== Price History for {company} ===\n"
                f"Current Price: ${price_summary['current_price']}\n"
                f"1-Month Change: {price_summary['1m_change_pct']}%\n"
                f"Period Change: {price_summary['period_change_pct']}%\n"
                f"Period High: ${price_summary['period_high']}\n"
                f"Period Low: ${price_summary['period_low']}\n"
                f"Avg Daily Volume: {price_summary['avg_volume']:,}"
            )

            return CollectedData(
                source="yahoo_finance",
                data_type="financial",
                content=price_summary,
                raw_text=text,
                metadata={"sub_type": "price_history", "ticker": company},
            )
        except Exception as e:
            logger.warning(f"Failed to extract price history: {e}")
            return None

    def _extract_recommendations(
        self, ticker_data: dict, company: str
    ) -> CollectedData | None:
        """Extract analyst recommendations."""
        recs = ticker_data.get("recommendations")
        if recs is None or (isinstance(recs, pd.DataFrame) and recs.empty):
            return None

        try:
            if isinstance(recs, pd.DataFrame):
                recent = recs.tail(10)
                rec_text = f"=== Recent Analyst Recommendations for {company} ===\n"
                rec_text += recent.to_string()

                return CollectedData(
                    source="yahoo_finance",
                    data_type="financial",
                    content={"recommendations": recent.to_dict()},
                    raw_text=rec_text,
                    metadata={
                        "sub_type": "recommendations",
                        "ticker": company,
                    },
                    reliability_score=0.7,
                )
        except Exception as e:
            logger.warning(f"Failed to extract recommendations: {e}")

        return None
