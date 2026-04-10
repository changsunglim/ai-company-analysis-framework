"""
Financial data collector - uses yfinance for metrics, prices, etc.
"""

import asyncio
from typing import Any

import yfinance as yf
import pandas as pd

from src.collector.base import BaseCollector, CollectedData
from src.utils.logger import setup_logger

logger = setup_logger("financial_collector")


class FinancialCollector(BaseCollector):
    """Collects financial data from Yahoo Finance."""

    DEFAULT_METRICS = [
        "revenue", "net_income", "operating_margin",
        "debt_to_equity", "free_cash_flow", "pe_ratio", "market_cap",
    ]

    async def collect(self, company: str, **kwargs) -> list[CollectedData]:
        """Collect financial data for a ticker symbol."""
        logger.info(f"Fetching financial data: {company}")

        # yfinance는 sync라서 executor로 돌림
        loop = asyncio.get_event_loop()
        ticker_data = await loop.run_in_executor(
            None, self._fetch_ticker_data, company
        )

        if not ticker_data:
            logger.warning(f"No data for {company}")
            return []

        results = []

        profile = self._extract_profile(ticker_data)
        if profile:
            results.append(profile)

        metrics = self._extract_metrics(ticker_data, company)
        if metrics:
            results.append(metrics)

        history = self._extract_price_history(ticker_data, company)
        if history:
            results.append(history)

        recs = self._extract_recommendations(ticker_data, company)
        if recs:
            results.append(recs)

        logger.info(f"Got {len(results)} data points for {company}")
        return results

    def _fetch_ticker_data(self, ticker: str) -> dict[str, Any] | None:
        """Fetch everything from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or info.get("trailingPegRatio") is None and not info.get("longName"):
                pass  # might be invalid ticker but let's try anyway

            data = {"info": info, "stock": stock}

            # 각각 따로 try/except 해야 하나가 실패해도 나머지는 됨
            try:
                data["income_stmt"] = stock.income_stmt
            except Exception:
                data["income_stmt"] = None

            try:
                data["balance_sheet"] = stock.balance_sheet
            except Exception:
                data["balance_sheet"] = None

            try:
                data["history"] = stock.history(period="1y")
            except Exception:
                data["history"] = None

            try:
                data["recommendations"] = stock.recommendations
            except Exception:
                data["recommendations"] = None

            return data

        except Exception as e:
            logger.error(f"yfinance failed for {ticker}: {e}")
            return None

    def _extract_profile(self, ticker_data: dict) -> CollectedData | None:
        info = ticker_data.get("info", {})
        if not info:
            return None

        text = (
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
            raw_text=text,
            metadata={"sub_type": "profile"},
        )

    def _extract_metrics(self, ticker_data: dict, company: str) -> CollectedData | None:
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

        lines = [f"=== Financial Metrics: {company} ===\n"]
        for key, val in metrics.items():
            if val is None:
                continue
            label = key.replace("_", " ").title()
            if isinstance(val, float):
                if "margin" in key or "growth" in key or "return" in key or "yield" in key:
                    lines.append(f"{label}: {val:.2%}")
                else:
                    lines.append(f"{label}: {val:,.2f}")
            elif isinstance(val, int):
                lines.append(f"{label}: {val:,}")
            else:
                lines.append(f"{label}: {val}")

        return CollectedData(
            source="yahoo_finance",
            data_type="financial",
            content=metrics,
            raw_text="\n".join(lines),
            metadata={"sub_type": "metrics", "ticker": company},
        )

    def _extract_price_history(self, ticker_data: dict, company: str) -> CollectedData | None:
        history = ticker_data.get("history")
        if history is None or history.empty:
            return None

        try:
            current = history["Close"].iloc[-1]
            month_ago = history["Close"].iloc[-22] if len(history) > 22 else history["Close"].iloc[0]
            year_start = history["Close"].iloc[0]

            summary = {
                "current_price": round(current, 2),
                "1m_change_pct": round((current - month_ago) / month_ago * 100, 2),
                "period_change_pct": round((current - year_start) / year_start * 100, 2),
                "period_high": round(history["Close"].max(), 2),
                "period_low": round(history["Close"].min(), 2),
                "avg_volume": int(history["Volume"].mean()),
            }

            text = (
                f"=== Price History: {company} ===\n"
                f"Current: ${summary['current_price']}\n"
                f"1M Change: {summary['1m_change_pct']}%\n"
                f"Period Change: {summary['period_change_pct']}%\n"
                f"High: ${summary['period_high']}\n"
                f"Low: ${summary['period_low']}\n"
                f"Avg Volume: {summary['avg_volume']:,}"
            )

            return CollectedData(
                source="yahoo_finance",
                data_type="financial",
                content=summary,
                raw_text=text,
                metadata={"sub_type": "price_history", "ticker": company},
            )
        except Exception as e:
            logger.warning(f"Price history extraction failed: {e}")
            return None

    def _extract_recommendations(self, ticker_data: dict, company: str) -> CollectedData | None:
        recs = ticker_data.get("recommendations")
        if recs is None or (isinstance(recs, pd.DataFrame) and recs.empty):
            return None

        try:
            if isinstance(recs, pd.DataFrame):
                recent = recs.tail(10)
                text = f"=== Analyst Recs: {company} ===\n" + recent.to_string()

                return CollectedData(
                    source="yahoo_finance",
                    data_type="financial",
                    content={"recommendations": recent.to_dict()},
                    raw_text=text,
                    metadata={"sub_type": "recommendations", "ticker": company},
                    reliability_score=0.7,
                )
        except Exception as e:
            logger.warning(f"Recommendations failed: {e}")

        return None
