"""
News data collector using web scraping and RSS feeds.

Gathers recent news articles and headlines related to the target company.
"""

import asyncio
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import aiohttp
from bs4 import BeautifulSoup

from src.collector.base import BaseCollector, CollectedData
from src.utils.logger import setup_logger

logger = setup_logger("news_collector")


class NewsCollector(BaseCollector):
    """
    Collects news articles from Google News RSS.

    Uses publicly available RSS feeds to gather recent headlines
    and article snippets without requiring paid API keys.
    """

    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.max_articles = self.config.get("max_articles", 15)
        self.lookback_days = self.config.get("lookback_days", 30)

    async def collect(
        self, company: str, **kwargs
    ) -> list[CollectedData]:
        """
        Collect news articles for a company.

        Args:
            company: Company name to search for

        Returns:
            List of CollectedData containing news articles
        """
        logger.info(f"Collecting news for: {company}")

        articles = await self._fetch_google_news(company)

        if not articles:
            logger.warning(f"No news articles found for {company}")
            return []

        # Bundle articles into a single CollectedData
        all_articles_text = f"=== Recent News for {company} ===\n\n"
        article_list = []

        for i, article in enumerate(articles[: self.max_articles], 1):
            all_articles_text += (
                f"--- Article {i} ---\n"
                f"Title: {article['title']}\n"
                f"Source: {article['source']}\n"
                f"Date: {article['published']}\n"
                f"Summary: {article['summary']}\n\n"
            )
            article_list.append(article)

        result = CollectedData(
            source="google_news",
            data_type="news",
            content={"articles": article_list, "count": len(article_list)},
            raw_text=all_articles_text,
            metadata={
                "query": company,
                "article_count": len(article_list),
            },
            reliability_score=0.6,  # News requires LLM verification
        )

        logger.info(f"Collected {len(article_list)} news articles")
        return [result]

    async def _fetch_google_news(
        self, query: str
    ) -> list[dict]:
        """Fetch news articles from Google News RSS feed."""
        url = self.GOOGLE_NEWS_RSS.format(query=quote_plus(query))
        articles = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15),
                    headers={"User-Agent": "Mozilla/5.0"},
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Google News returned status {response.status}"
                        )
                        return []

                    content = await response.text()

            soup = BeautifulSoup(content, "html.parser")
            items = soup.find_all("item")

            cutoff_date = datetime.now() - timedelta(
                days=self.lookback_days
            )

            for item in items:
                try:
                    title = item.title.text if item.title else ""
                    link = item.link.text if item.link else ""
                    pub_date = item.pubdate.text if item.pubdate else ""
                    description = item.description.text if item.description else ""

                    # Clean HTML from description
                    if description:
                        desc_soup = BeautifulSoup(description, "html.parser")
                        description = desc_soup.get_text(separator=" ").strip()

                    # Extract source from title (Google News format: "Title - Source")
                    source = ""
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        title = parts[0].strip()
                        source = parts[1].strip()

                    articles.append(
                        {
                            "title": title,
                            "source": source,
                            "url": link,
                            "published": pub_date,
                            "summary": description[:500],
                        }
                    )

                except Exception as e:
                    logger.debug(f"Failed to parse news item: {e}")
                    continue

        except asyncio.TimeoutError:
            logger.warning("Google News request timed out")
        except Exception as e:
            logger.error(f"Failed to fetch Google News: {e}")

        return articles
