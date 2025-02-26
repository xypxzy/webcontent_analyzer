import asyncio
from typing import Dict, Any, Optional, List, Set, Tuple
from urllib.parse import urlparse, urljoin
import random
import time
from datetime import datetime, timedelta
import hashlib
import re

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

from app.core.config import settings
from app.services.parser.html_extractor import (
    extract_main_content,
    extract_text_content,
    extract_structured_content,
)
from app.services.parser.metadata_extractor import (
    extract_metadata,
    extract_microdata,
    # extract_opengraph,
)
from app.services.parser.dynamic_content_processor import DynamicContentProcessor
from app.services.parser.robots_checker import RobotsChecker
from app.services.parser.content_cache import ContentCache
from app.services.parser.structure_analyzer import StructureAnalyzer

# Common user agent strings for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]


class PageParser:
    def __init__(self):
        self.timeout = settings.PARSER_TIMEOUT
        self.max_retries = settings.PARSER_MAX_RETRIES
        self.requests_per_second = settings.PARSER_REQUESTS_PER_SECOND
        self.default_user_agent = settings.PARSER_USER_AGENT

        # Initialize helper components
        self.dynamic_processor = DynamicContentProcessor()
        self.robots_checker = RobotsChecker()
        self.content_cache = ContentCache()
        self.structure_analyzer = StructureAnalyzer()

        # Request throttling
        self.domain_request_times = {}  # Track last request time per domain
        self.domain_locks = {}  # Domain-specific locks for concurrency control

        # Shared session lock
        self.session_lock = asyncio.Lock()

    async def parse_url(
        self, url: str, parse_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Parse a URL and extract content, structure, and metadata.

        Args:
            url: The URL to parse
            parse_options: Optional configuration for parsing behavior
                {
                    'render_js': bool,              # Whether to render JavaScript
                    'depth': int,                   # Depth for internal link following
                    'follow_links': bool,           # Whether to follow internal links
                    'extract_metadata': bool,       # Extract meta tags, microdata, etc.
                    'extract_structured_data': bool,# Extract structured content (tables, lists)
                    'ignore_robots': bool,          # Whether to ignore robots.txt
                    'cache_ttl': int,               # Cache time-to-live in seconds
                    'headers': Dict[str, str],      # Custom headers to pass
                    'proxy': str,                   # Optional proxy to use
                    'timeout': int,                 # Custom timeout
                    'user_interactions': List[Dict] # User interactions for dynamic pages
                }

        Returns:
            Dictionary containing parsed data
        """
        # Set default options if not provided
        if parse_options is None:
            parse_options = {}

        # Normalize URL and get domain
        url = self._normalize_url(url)
        domain = urlparse(url).netloc

        # Check cache before making any requests
        if not parse_options.get("ignore_cache", False):
            cached_content = await self.content_cache.get(url)
            if cached_content:
                logger.info(f"Using cached content for {url}")
                return cached_content

        # Check robots.txt unless explicitly ignored
        if not parse_options.get("ignore_robots", False):
            allowed = await self.robots_checker.check_url(url, self.default_user_agent)
            if not allowed:
                logger.warning(f"URL {url} is disallowed by robots.txt")
                return {"success": False, "error": "URL is disallowed by robots.txt"}

        # Get or create domain lock
        if domain not in self.domain_locks:
            self.domain_locks[domain] = asyncio.Lock()

        # Handle throttling per domain
        async with self.domain_locks[domain]:
            # Calculate delay since last request to this domain
            now = time.time()
            last_request_time = self.domain_request_times.get(domain, 0)
            delay = max(0, (1.0 / self.requests_per_second) - (now - last_request_time))

            if delay > 0:
                logger.debug(f"Throttling request to {domain}, waiting {delay:.2f}s")
                await asyncio.sleep(delay)

            # Update last request time
            self.domain_request_times[domain] = time.time()

        try:
            # Decide whether to use static or dynamic parsing
            if parse_options.get("render_js", False):
                html_content, is_dynamic = await self._fetch_with_js_rendering(
                    url, parse_options.get("user_interactions", [])
                )
            else:
                html_content, is_dynamic = await self._fetch_url(
                    url, options=parse_options
                )

            if not html_content:
                return {"success": False, "error": "Failed to fetch URL"}

            # Create BeautifulSoup object with appropriate parser
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract basic page information
            title = soup.title.string.strip() if soup.title else ""
            favicon = self._extract_favicon(soup, url)
            lang = self._extract_language(soup)

            # Extract metadata if requested
            metadata = {}
            if parse_options.get("extract_metadata", True):
                metadata = extract_metadata(soup, url)
                if parse_options.get("include_microdata", True):
                    metadata["microdata"] = extract_microdata(soup)
                # if parse_options.get("include_opengraph", True):
                # metadata["opengraph"] = extract_opengraph(soup)

            # Extract main content and text
            main_content = extract_main_content(soup)
            text_content = extract_text_content(main_content)

            # Extract structured content if requested
            structured_content = {}
            if parse_options.get("extract_structured_data", True):
                structured_content = extract_structured_content(soup)

            # Analyze page structure
            structure = self.structure_analyzer.analyze(soup, url)

            # Collect links for potential further processing
            links = self._extract_links(soup, url)

            # Construct result
            result = {
                "success": True,
                "url": url,
                "title": title,
                "html_content": html_content,
                "main_content": str(main_content),
                "text_content": text_content,
                "structured_content": structured_content,
                "page_metadata": metadata,
                "structure": structure,
                "is_dynamic": is_dynamic,
                "favicon": favicon,
                "language": lang,
                "links": links,
                "parsed_at": datetime.utcnow().isoformat(),
            }

            # Cache the result
            cache_ttl = parse_options.get("cache_ttl", 3600)  # Default 1 hour
            await self.content_cache.set(url, result, ttl=cache_ttl)

            # Follow links if requested (for sitemap building)
            if (
                parse_options.get("follow_links", False)
                and parse_options.get("depth", 0) > 0
            ):
                await self._follow_links(links, parse_options)

            return result

        except Exception as e:
            logger.error(f"Error parsing URL {url}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _fetch_url(
        self, url: str, retry_count: int = 0, options: Dict[str, Any] = None
    ) -> Tuple[Optional[str], bool]:
        """
        Fetch URL content with retry logic.

        Args:
            url: URL to fetch
            retry_count: Current retry attempt
            options: Additional options for the request

        Returns:
            Tuple of (HTML content as string or None if failed, is_dynamic flag)
        """
        if options is None:
            options = {}

        # Prepare headers with rotating user agent
        user_agent = options.get(
            "user_agent",
            random.choice(USER_AGENTS) if USER_AGENTS else self.default_user_agent,
        )
        timeout = options.get("timeout", self.timeout)

        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

        # Add custom headers if provided
        if "headers" in options:
            headers.update(options["headers"])

        try:
            # Create a client session with proxy if provided
            proxy = options.get("proxy")

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=True,
                    proxy=proxy,
                    ssl=options.get("verify_ssl", True),
                ) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Try to detect if page requires JavaScript
                        is_dynamic = self._is_dynamic_page(content)

                        # If dynamic content is detected but JS rendering wasn't requested,
                        # we'll still return the static content but flag it
                        return content, is_dynamic

                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")

                    # Handle specific HTTP errors
                    if response.status == 403:
                        return None, False  # Forbidden
                    elif response.status == 404:
                        return None, False  # Not found
                    elif response.status == 429:
                        # Too many requests - wait longer before retry
                        retry_delay = int(response.headers.get("Retry-After", 30))
                        await asyncio.sleep(retry_delay)
                        if retry_count < self.max_retries:
                            return await self._fetch_url(url, retry_count + 1, options)
                    elif response.status >= 500:
                        # Server error, retry with backoff
                        if retry_count < self.max_retries:
                            await asyncio.sleep(2**retry_count)  # Exponential backoff
                            return await self._fetch_url(url, retry_count + 1, options)

                    return None, False

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if retry_count < self.max_retries:
                wait_time = 2**retry_count  # Exponential backoff
                logger.info(
                    f"Retrying {url} in {wait_time} seconds (attempt {retry_count + 1})"
                )
                await asyncio.sleep(wait_time)
                return await self._fetch_url(url, retry_count + 1, options)

            logger.error(
                f"Failed to fetch {url} after {self.max_retries} retries: {str(e)}"
            )
            return None, False

    async def _fetch_with_js_rendering(
        self, url: str, user_interactions: List[Dict] = None
    ) -> Tuple[Optional[str], bool]:
        """
        Fetch URL content with JavaScript rendering support.

        Args:
            url: URL to render
            user_interactions: List of user interactions to perform
                [
                    {'type': 'click', 'selector': '.button'},
                    {'type': 'scroll', 'amount': 800},
                    {'type': 'wait', 'time': 2},
                    {'type': 'input', 'selector': '#search', 'value': 'query'}
                ]

        Returns:
            Tuple of (rendered HTML content, is_dynamic flag)
        """
        try:
            html_content = await self.dynamic_processor.render_page(
                url, user_interactions=user_interactions
            )
            return html_content, True
        except Exception as e:
            logger.error(f"Error rendering JavaScript for {url}: {str(e)}")

            # Fall back to regular fetching if JS rendering fails
            logger.info(f"Falling back to regular HTTP request for {url}")
            html_content, _ = await self._fetch_url(url)
            return html_content, False

    def _is_dynamic_page(self, html_content: str) -> bool:
        """
        Detect if a page likely requires JavaScript for rendering.

        Args:
            html_content: HTML content to analyze

        Returns:
            Boolean indicating if page likely requires JS rendering
        """
        # Check for SPA frameworks
        framework_patterns = [
            r"ng-app",  # Angular
            r"react-app",  # React
            r"data-vue",  # Vue.js
            r"ember-application",  # Ember
            r"<nuxt",  # Nuxt.js
            r"__NEXT_DATA__",  # Next.js
            r"ng-controller",  # Angular
            r"v-if|v-for|v-model|v-on",  # Vue.js directives
        ]

        for pattern in framework_patterns:
            if re.search(pattern, html_content):
                return True

        # Check for AJAX load patterns
        ajax_patterns = [
            r"\.load\(",
            r"\.ajax\(",
            r"fetch\(",
            r"new XMLHttpRequest\(\)",
            r"axios",
        ]

        for pattern in ajax_patterns:
            if re.search(pattern, html_content):
                return True

        # Check for loading indicators or empty content containers
        loading_patterns = [
            r'class="[^"]*loading[^"]*"',
            r'id="[^"]*loading[^"]*"',
            r"skeleton|placeholder|lazy-load",
        ]

        for pattern in loading_patterns:
            if re.search(pattern, html_content):
                return True

        # Check the ratio of script tags to content
        soup = BeautifulSoup(html_content, "html.parser")
        scripts = soup.find_all("script")
        content_elements = soup.find_all(["p", "div", "article", "section"])

        # If there are many scripts but little content, it may be a JS-heavy page
        if len(scripts) > 10 and len(content_elements) < 5:
            return True

        return False

    def _extract_links(
        self, soup: BeautifulSoup, base_url: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract internal and external links from the page.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links

        Returns:
            Dictionary with internal and external links
        """
        internal_links = []
        external_links = []
        base_domain = urlparse(base_url).netloc

        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href", "").strip()

            # Skip empty, javascript, and anchor links
            if not href or href.startswith(("javascript:", "#", "mailto:", "tel:")):
                continue

            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            parsed_url = urlparse(full_url)

            # Create link object
            link_obj = {
                "url": full_url,
                "text": a_tag.get_text().strip(),
                "rel": a_tag.get("rel", ""),
                "title": a_tag.get("title", ""),
                "nofollow": "nofollow" in a_tag.get("rel", []),
            }

            # Check if internal or external
            if parsed_url.netloc == base_domain or not parsed_url.netloc:
                internal_links.append(link_obj)
            else:
                external_links.append(link_obj)

        return {"internal": internal_links, "external": external_links}

    async def _follow_links(
        self, links: Dict[str, List[Dict]], options: Dict[str, Any]
    ) -> None:
        """
        Asynchronously follow internal links for sitemap building.

        Args:
            links: Dictionary with internal and external links
            options: Parsing options including depth
        """
        # Decrease depth by 1 for recursive calls
        depth = options.get("depth", 0) - 1
        if depth <= 0:
            return

        # Update options for child requests
        child_options = options.copy()
        child_options["depth"] = depth
        child_options["follow_links"] = True

        # Process only internal links
        internal_links = links.get("internal", [])

        # Limit the number of links to follow
        max_links = options.get("max_links_to_follow", 10)
        internal_links = internal_links[:max_links]

        # Create tasks for each link
        tasks = []
        for link in internal_links:
            url = link["url"]
            tasks.append(self.parse_url(url, parse_options=child_options))

        # Execute tasks with concurrency limit
        concurrency_limit = options.get("concurrency_limit", 5)
        for i in range(0, len(tasks), concurrency_limit):
            batch = tasks[i : i + concurrency_limit]
            await asyncio.gather(*batch)

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing fragments and standardizing format.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        # Parse URL
        parsed = urlparse(url)

        # Remove fragment
        normalized = parsed._replace(fragment="")

        # Ensure scheme
        if not normalized.scheme:
            normalized = normalized._replace(scheme="http")

        return normalized.geturl()

    def _extract_favicon(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """
        Extract favicon URL from the page.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links

        Returns:
            Favicon URL or None
        """
        # Check for link rel="icon" or rel="shortcut icon"
        for link in soup.find_all(
            "link", rel=lambda r: r and ("icon" in r or "shortcut icon" in r)
        ):
            if "href" in link.attrs:
                return urljoin(base_url, link["href"])

        # Default to /favicon.ico if nothing found
        return urljoin(base_url, "/favicon.ico")

    def _extract_language(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract language from the HTML document.

        Args:
            soup: BeautifulSoup object

        Returns:
            Language code or None
        """
        # Check html tag lang attribute
        html_tag = soup.find("html")
        if html_tag and "lang" in html_tag.attrs:
            return html_tag["lang"]

        # Check meta tags
        meta_lang = soup.find("meta", attrs={"http-equiv": "content-language"})
        if meta_lang and "content" in meta_lang.attrs:
            return meta_lang["content"]

        return None
