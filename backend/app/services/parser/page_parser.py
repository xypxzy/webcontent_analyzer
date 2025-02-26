import asyncio
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

from app.core.config import settings
from app.services.parser.html_extractor import (
    extract_main_content,
    extract_text_content,
)
from app.services.parser.metadata_extractor import extract_metadata


class PageParser:
    def __init__(self):
        self.user_agent = settings.PARSER_USER_AGENT
        self.timeout = settings.PARSER_TIMEOUT
        self.max_retries = settings.PARSER_MAX_RETRIES

    async def parse_url(self, url: str) -> Dict[str, Any]:
        """
        Parse a URL and extract content, structure, and metadata.

        Args:
            url: The URL to parse

        Returns:
            Dictionary containing parsed data
        """
        try:
            html_content = await self._fetch_url(url)
            if not html_content:
                return {"success": False, "error": "Failed to fetch URL"}

            # Create BeautifulSoup object
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract various components
            title = soup.title.string if soup.title else ""
            metadata = extract_metadata(soup, url)
            main_content = extract_main_content(soup)
            text_content = extract_text_content(main_content)

            # Build a structure of the page
            structure = self._analyze_structure(soup)

            return {
                "success": True,
                "url": url,
                "title": title,
                "html_content": html_content,
                "main_content": str(main_content),
                "text_content": text_content,
                "metadata": metadata,
                "structure": structure,
            }

        except Exception as e:
            logger.error(f"Error parsing URL {url}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _fetch_url(self, url: str, retry_count: int = 0) -> Optional[str]:
        """
        Fetch URL content with retry logic.

        Args:
            url: URL to fetch
            retry_count: Current retry attempt

        Returns:
            HTML content as string or None if failed
        """
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, timeout=self.timeout, allow_redirects=True
                ) as response:
                    if response.status == 200:
                        return await response.text()
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if retry_count < self.max_retries:
                wait_time = 2**retry_count  # Exponential backoff
                logger.info(
                    f"Retrying {url} in {wait_time} seconds (attempt {retry_count + 1})"
                )
                await asyncio.sleep(wait_time)
                return await self._fetch_url(url, retry_count + 1)
            logger.error(
                f"Failed to fetch {url} after {self.max_retries} retries: {str(e)}"
            )
            return None

    def _analyze_structure(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze the structure of the HTML document.

        Args:
            soup: BeautifulSoup object of the HTML

        Returns:
            Dictionary with structure information
        """
        # Collect headings
        headings = {
            "h1": [h.get_text().strip() for h in soup.find_all("h1")],
            "h2": [h.get_text().strip() for h in soup.find_all("h2")],
            "h3": [h.get_text().strip() for h in soup.find_all("h3")],
        }

        # Collect links
        links = [
            {"text": a.get_text().strip(), "href": a.get("href", "")}
            for a in soup.find_all("a")
            if a.get("href")
        ]

        # Find CTA elements (buttons, forms)
        cta_elements = []
        for button in soup.find_all("button"):
            cta_elements.append(
                {
                    "type": "button",
                    "text": button.get_text().strip(),
                    "classes": button.get("class", []),
                }
            )

        for form in soup.find_all("form"):
            cta_elements.append(
                {
                    "type": "form",
                    "action": form.get("action", ""),
                    "method": form.get("method", "get"),
                    "fields": [
                        {"name": inp.get("name", ""), "type": inp.get("type", "")}
                        for inp in form.find_all(["input", "textarea", "select"])
                    ],
                }
            )

        # Collect images
        images = [
            {"src": img.get("src", ""), "alt": img.get("alt", "")}
            for img in soup.find_all("img")
            if img.get("src")
        ]

        return {
            "headings": headings,
            "links": links,
            "cta_elements": cta_elements,
            "images": images,
            "paragraphs_count": len(soup.find_all("p")),
            "lists": {
                "ul": len(soup.find_all("ul")),
                "ol": len(soup.find_all("ol")),
            },
        }
