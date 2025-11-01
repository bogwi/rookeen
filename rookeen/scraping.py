from __future__ import annotations

import asyncio
import re
import time
from types import TracebackType
from urllib.robotparser import RobotFileParser

import aiohttp
from bs4 import BeautifulSoup

from rookeen.models import WebPageContent
from rookeen.utils.robots import politeness_delay


class AsyncWebScraper:
    """Asynchronous web scraper for content extraction.

    Uses aiohttp for fetching and BeautifulSoup for sanitization.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit: float = 0.5,
        robots_policy: str = "respect",
    ):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: aiohttp.ClientSession | None = None
        self.rate_limit = rate_limit
        self.robots_policy = robots_policy
        self.robots_parser: RobotFileParser | None = None
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

    async def __aenter__(self) -> AsyncWebScraper:
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers=self.headers,
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5),
        )
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        if self.session:
            await self.session.close()

    async def fetch_page(self, url: str) -> WebPageContent:
        """Fetch and parse web page content with basic retries."""
        if not self.session:
            raise RuntimeError("AsyncWebScraper not properly initialized")

        # Check robots.txt before attempting to fetch
        if not self._check_robots_txt(url):
            raise aiohttp.ClientError(f"Robots.txt disallows crawling: {url}")

        # Apply rate limiting
        politeness_delay(url, self.rate_limit)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                await asyncio.sleep(0.5 * attempt)  # polite backoff
                async with self.session.get(url) as response:
                    if response.status != 200:
                        raise aiohttp.ClientError(f"HTTP {response.status}: {response.reason}")

                    html_content = await response.text()
                    text_content = self._extract_text(html_content)
                    title = self._extract_title(html_content)

                    return WebPageContent(
                        url=url,
                        title=title,
                        text=text_content,
                        html=html_content,
                        timestamp=time.time(),
                        word_count=len(text_content.split()),
                        char_count=len(text_content),
                    )
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise

        # Should not reach here
        assert last_exc is not None
        raise last_exc

    def _extract_text(self, html: str) -> str:
        """Extract clean text from HTML using BeautifulSoup; regex fallback."""
        text = ""
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
        except Exception:
            # Regex fallback (less robust)
            cleaned = re.sub(
                r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
            )
            cleaned = re.sub(
                r"<style[^>]*>.*?</style>", "", cleaned, flags=re.DOTALL | re.IGNORECASE
            )
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)
            text = re.sub(r"\s+", " ", cleaned).strip()
        return text

    def _extract_title(self, html: str) -> str:
        """Extract page title from HTML."""
        try:
            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.string:
                title_string: str = soup.title.string.strip()
                return title_string
        except Exception:
            pass
        # Regex fallback
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return "Untitled"

    def _check_robots_txt(self, url: str) -> bool:
        """Check if crawling is allowed by robots.txt."""
        if self.robots_policy == "ignore":
            return True

        if self.robots_parser is None:
            from urllib.parse import urljoin

            robots_url = urljoin(url, "/robots.txt")
            self.robots_parser = RobotFileParser()
            try:
                self.robots_parser.set_url(robots_url)
                self.robots_parser.read()
            except Exception:
                # If robots.txt can't be fetched, assume allowed
                return True

        user_agent = self.headers.get("User-Agent", "*")
        return self.robots_parser.can_fetch(user_agent, url)
