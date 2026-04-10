"""
News collector - scrapes Google News RSS for recent articles.
No API key needed which is nice.
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
    """Collects news from Google News RSS feed."""

    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.max_articles = self.config.get("max_articles", 15)
        self.lookback_days = self.config.get("lookback_days", 30)

    async def collect(self, company: str, **kwargs) -> list[CollectedData]:
        logger.info(f"Fetching news: {company}")

        articles = await self._fetch_google_news(company)

        if not articles:
            logger.warning(f"No news found for {company}")
            return []

        # 기사 전부 하나로 묶어서 반환
        text = f"=== Recent News: {company} ===\n\n"
        article_list = []

        for i, article in enumerate(articles[:self.max_articles], 1):
            text += (
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
            raw_text=text,
            metadata={"query": company, "article_count": len(article_list)},
            reliability_score=0.6,  # 뉴스는 LLM이 검증해야됨
        )

        logger.info(f"Got {len(article_list)} articles")
        return [result]

    async def _fetch_google_news(self, query: str) -> list[dict]:
        url = self.GOOGLE_NEWS_RSS.format(query=quote_plus(query))
        articles = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15),
                    headers={"User-Agent": "Mozilla/5.0"},
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Google News status: {resp.status}")
                        return []
                    content = await resp.text()

            soup = BeautifulSoup(content, "html.parser")
            items = soup.find_all("item")

            cutoff = datetime.now() - timedelta(days=self.lookback_days)

            for item in items:
                try:
                    title = item.title.text if item.title else ""
                    link = item.link.text if item.link else ""
                    pub_date = item.pubdate.text if item.pubdate else ""
                    desc = item.description.text if item.description else ""

                    # description에서 HTML 태그 제거
                    if desc:
                        desc_soup = BeautifulSoup(desc, "html.parser")
                        desc = desc_soup.get_text(separator=" ").strip()

                    # Google News는 "Title - Source" 형식임
                    source = ""
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        title = parts[0].strip()
                        source = parts[1].strip()

                    articles.append({
                        "title": title,
                        "source": source,
                        "url": link,
                        "published": pub_date,
                        "summary": desc[:500],  # 너무 길면 잘라냄
                    })

                except Exception as e:
                    logger.debug(f"Failed to parse item: {e}")
                    continue

        except asyncio.TimeoutError:
            logger.warning("Google News timed out")
        except Exception as e:
            logger.error(f"Google News fetch failed: {e}")

        return articles
