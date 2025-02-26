import asyncio
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import uuid
import os
import json
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from loguru import logger

from app.services.parser.page_parser import PageParser
from app.services.parser.robots_checker import RobotsChecker
from app.services.parser.content_cache import ContentCache
from app.services.parser.dynamic_content_processor import DynamicContentProcessor
from app.services.parser.structure_analyzer import StructureAnalyzer


@dataclass
class ParseOptions:
    """Options for page parsing."""

    # Basic options
    render_js: bool = False
    extract_metadata: bool = True
    extract_structured_data: bool = True
    ignore_robots: bool = False

    # Cache settings
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 hour

    # Link crawling options
    follow_links: bool = False
    depth: int = 0
    max_links_to_follow: int = 10

    # Advanced options
    user_agent: Optional[str] = None
    timeout: Optional[int] = None
    proxy: Optional[str] = None
    verify_ssl: bool = True
    headers: Dict[str, str] = field(default_factory=dict)

    # Dynamic content options
    user_interactions: List[Dict[str, Any]] = field(default_factory=list)
    wait_for_selector: Optional[str] = None

    # Processing options
    main_content_only: bool = True
    extract_text: bool = True
    detect_duplicates: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert options to dictionary."""
        return {
            "render_js": self.render_js,
            "extract_metadata": self.extract_metadata,
            "extract_structured_data": self.extract_structured_data,
            "ignore_robots": self.ignore_robots,
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl,
            "follow_links": self.follow_links,
            "depth": self.depth,
            "max_links_to_follow": self.max_links_to_follow,
            "user_agent": self.user_agent,
            "timeout": self.timeout,
            "proxy": self.proxy,
            "verify_ssl": self.verify_ssl,
            "headers": self.headers,
            "user_interactions": self.user_interactions,
            "wait_for_selector": self.wait_for_selector,
            "main_content_only": self.main_content_only,
            "extract_text": self.extract_text,
            "detect_duplicates": self.detect_duplicates,
        }


class WebContentAnalyzer:
    """
    Main class for analyzing web content.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the web content analyzer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        self.parser = PageParser()
        self.robots_checker = RobotsChecker()
        self.content_cache = ContentCache(
            use_redis=self.config.get("use_redis", False),
            redis_url=self.config.get("redis_url"),
        )
        self.dynamic_processor = DynamicContentProcessor()
        self.structure_analyzer = StructureAnalyzer()

        # Initialize session data
        self.session_id = str(uuid.uuid4())
        self.processed_urls = set()
        self.current_requests = 0
        self.max_concurrent_requests = self.config.get("max_concurrent_requests", 10)
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        logger.info(
            f"WebContentAnalyzer initialized with session ID: {self.session_id}"
        )

    async def analyze_url(
        self, url: str, options: Optional[ParseOptions] = None
    ) -> Dict[str, Any]:
        """
        Analyze a URL by parsing and extracting structured information.

        Args:
            url: URL to analyze
            options: Parsing options

        Returns:
            Dictionary with analysis results
        """
        # Use default options if not provided
        if options is None:
            options = ParseOptions()

        # Convert to dictionary for parsing
        parse_options = options.to_dict()

        # Add session tracking
        self.processed_urls.add(url)

        # Acquire semaphore to limit concurrent requests
        async with self.request_semaphore:
            # Parse URL and get content
            parsed_data = await self.parser.parse_url(url, parse_options)

            # Handle parsing failure
            if not parsed_data.get("success", False):
                return {
                    "success": False,
                    "error": parsed_data.get("error", "Unknown error during parsing"),
                    "url": url,
                    "session_id": self.session_id,
                }

            # Process the parsed data
            result = await self._process_parsed_data(url, parsed_data, options)

            # Add session info
            result["session_id"] = self.session_id

            return result

    async def analyze_html(
        self,
        html: str,
        url: str = "http://example.com",
        options: Optional[ParseOptions] = None,
    ) -> Dict[str, Any]:
        """
        Analyze HTML content directly.

        Args:
            html: HTML content to analyze
            url: Base URL for resolving relative links
            options: Parsing options

        Returns:
            Dictionary with analysis results
        """
        # Use default options if not provided
        if options is None:
            options = ParseOptions()

        # Create base parsed data structure
        parsed_data = {
            "success": True,
            "url": url,
            "html_content": html,
            "title": "",
            "is_dynamic": False,
            "parsed_at": "",  # Will be filled in by parser
        }

        # Extract title from HTML
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        if title_tag:
            parsed_data["title"] = title_tag.get_text().strip()

        # Process the data
        result = await self._process_parsed_data(url, parsed_data, options)

        # Add session info
        result["session_id"] = self.session_id

        return result

    async def _process_parsed_data(
        self, url: str, parsed_data: Dict[str, Any], options: ParseOptions
    ) -> Dict[str, Any]:
        """
        Process parsed data to extract analysis results.

        Args:
            url: Source URL
            parsed_data: Data from parser
            options: Parsing options

        Returns:
            Processed analysis results
        """
        html_content = parsed_data.get("html_content", "")

        # Create BS object from HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Base result structure
        result = {
            "success": True,
            "url": url,
            "title": parsed_data.get("title", ""),
            "is_dynamic": parsed_data.get("is_dynamic", False),
            "parse_date": parsed_data.get("parsed_at", ""),
        }

        # Only include HTML content if requested
        if not options.main_content_only:
            result["html_content"] = html_content

        # Extract main content if requested
        if options.main_content_only and html_content:
            main_content = parsed_data.get("main_content")
            if not main_content:
                # Extract main content if not already done
                main_content_elem = self.parser.html_extractor.extract_main_content(
                    soup
                )
                main_content = str(main_content_elem)

            result["main_content"] = main_content

            # Create a separate soup for the main content
            content_soup = BeautifulSoup(main_content, "html.parser")
        else:
            content_soup = soup

        # Extract text if requested
        if options.extract_text and html_content:
            text_content = parsed_data.get("text_content")
            if not text_content:
                # Extract text if not already done
                if options.main_content_only:
                    text_content = self.parser.html_extractor.extract_text_content(
                        content_soup
                    )
                else:
                    text_content = self.parser.html_extractor.extract_text_content(soup)

            result["text"] = text_content

        # Extract metadata if requested
        if options.extract_metadata and html_content:
            metadata = parsed_data.get("page_metadata")
            if not metadata:
                # Extract metadata if not already done
                metadata = self.parser.html_extractor.extract_metadata(soup, url)

            result["metadata"] = metadata

        # Extract structured data if requested
        if options.extract_structured_data and html_content:
            structured_content = parsed_data.get("structured_content")
            if not structured_content:
                # Extract structured content if not already done
                target_soup = content_soup if options.main_content_only else soup
                structured_content = (
                    self.parser.html_extractor.extract_structured_content(target_soup)
                )

            result["structured_content"] = structured_content

        # Include structure analysis
        if html_content:
            structure = parsed_data.get("structure")
            if not structure:
                # Perform structure analysis if not already done
                structure = self.structure_analyzer.analyze(soup, url)

            result["structure"] = structure

        # Include link information
        if parsed_data.get("links"):
            result["links"] = parsed_data["links"]

        # Return the processed result
        return result

    async def batch_analyze(
        self, urls: List[str], options: Optional[ParseOptions] = None
    ) -> Dict[str, Any]:
        """
        Analyze multiple URLs in parallel.

        Args:
            urls: List of URLs to analyze
            options: Parsing options

        Returns:
            Dictionary with analysis results for each URL
        """
        # Use default options if not provided
        if options is None:
            options = ParseOptions()

        # Create tasks for each URL
        tasks = []
        for url in urls:
            task = self.analyze_url(url, options)
            tasks.append(task)

        # Run tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = {}

        for i, result in enumerate(results):
            url = urls[i]

            # Handle exceptions
            if isinstance(result, Exception):
                processed_results[url] = {
                    "success": False,
                    "error": str(result),
                    "url": url,
                    "session_id": self.session_id,
                }
            else:
                processed_results[url] = result

        return processed_results

    async def analyze_sitemap(
        self,
        sitemap_url: str,
        options: Optional[ParseOptions] = None,
        max_urls: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze a website using its sitemap.

        Args:
            sitemap_url: URL of the sitemap
            options: Parsing options
            max_urls: Maximum number of URLs to analyze

        Returns:
            Dictionary with analysis results
        """
        # Use default options if not provided
        if options is None:
            options = ParseOptions()

        # Parse the sitemap
        sitemap_data = await self._parse_sitemap(sitemap_url)

        if not sitemap_data.get("success", False):
            return {
                "success": False,
                "error": sitemap_data.get("error", "Failed to parse sitemap"),
                "sitemap_url": sitemap_url,
                "session_id": self.session_id,
            }

        # Get URLs from sitemap
        urls = sitemap_data.get("urls", [])

        # Limit the number of URLs
        if max_urls > 0:
            urls = urls[:max_urls]

        # Analyze the URLs
        if not urls:
            return {
                "success": True,
                "sitemap_url": sitemap_url,
                "message": "No URLs found in sitemap",
                "session_id": self.session_id,
                "results": {},
            }

        results = await self.batch_analyze(urls, options)

        return {
            "success": True,
            "sitemap_url": sitemap_url,
            "url_count": len(urls),
            "session_id": self.session_id,
            "results": results,
        }

    async def _parse_sitemap(self, sitemap_url: str) -> Dict[str, Any]:
        """
        Parse a sitemap to extract URLs.

        Args:
            sitemap_url: URL of the sitemap

        Returns:
            Dictionary with sitemap parsing results
        """
        try:
            # Parse the sitemap URL
            parse_options = {
                "extract_metadata": False,
                "extract_structured_data": False,
                "ignore_robots": True,
                "render_js": False,
            }

            parsed_data = await self.parser.parse_url(sitemap_url, parse_options)

            if not parsed_data.get("success", False):
                return {
                    "success": False,
                    "error": parsed_data.get("error", "Failed to fetch sitemap"),
                    "sitemap_url": sitemap_url,
                }

            # Parse XML
            soup = BeautifulSoup(parsed_data["html_content"], "xml")

            # Check if it's a sitemap index
            sitemaps = soup.find_all("sitemap")
            if sitemaps:
                # It's a sitemap index, get all sitemaps
                sitemap_urls = []
                for sitemap in sitemaps:
                    loc = sitemap.find("loc")
                    if loc:
                        sitemap_urls.append(loc.text)

                # Recursively parse all sitemaps
                all_urls = []
                for sm_url in sitemap_urls:
                    sm_result = await self._parse_sitemap(sm_url)
                    if sm_result.get("success", False):
                        all_urls.extend(sm_result.get("urls", []))

                return {
                    "success": True,
                    "urls": all_urls,
                    "sitemap_url": sitemap_url,
                    "is_index": True,
                    "child_sitemaps": sitemap_urls,
                }

            # Regular sitemap
            urls = []
            url_elements = soup.find_all("url")

            for url_elem in url_elements:
                loc = url_elem.find("loc")
                if loc:
                    urls.append(loc.text)

            return {
                "success": True,
                "urls": urls,
                "sitemap_url": sitemap_url,
                "is_index": False,
                "url_count": len(urls),
            }

        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {str(e)}")
            return {"success": False, "error": str(e), "sitemap_url": sitemap_url}

    async def crawl_website(
        self,
        start_url: str,
        options: Optional[ParseOptions] = None,
        max_pages: int = 100,
        max_depth: int = 3,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Crawl a website starting from a URL.

        Args:
            start_url: Starting URL
            options: Parsing options
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum crawl depth
            include_patterns: URL patterns to include (regex)
            exclude_patterns: URL patterns to exclude (regex)

        Returns:
            Dictionary with crawl results
        """
        # Use default options if not provided
        if options is None:
            options = ParseOptions()

        # Override options for crawling
        crawl_options = ParseOptions(
            render_js=options.render_js,
            extract_metadata=options.extract_metadata,
            extract_structured_data=options.extract_structured_data,
            ignore_robots=options.ignore_robots,
            use_cache=options.use_cache,
            cache_ttl=options.cache_ttl,
            follow_links=True,
            depth=max_depth,
            max_links_to_follow=max_pages,
            user_agent=options.user_agent,
            timeout=options.timeout,
            proxy=options.proxy,
            headers=options.headers,
            main_content_only=options.main_content_only,
            extract_text=options.extract_text,
        )

        # Initialize crawl state
        self.crawl_queue = [start_url]
        self.crawled_urls = set()
        self.crawl_results = {}

        # Create patterns
        include_compiled = [re.compile(p) for p in (include_patterns or [])]
        exclude_compiled = [re.compile(p) for p in (exclude_patterns or [])]

        # Get domain from start URL
        start_domain = urlparse(start_url).netloc

        # Start crawling
        while self.crawl_queue and len(self.crawled_urls) < max_pages:
            # Get next URL from queue
            url = self.crawl_queue.pop(0)

            # Skip if already crawled
            if url in self.crawled_urls:
                continue

            # Check patterns
            if include_compiled and not any(p.search(url) for p in include_compiled):
                continue

            if exclude_compiled and any(p.search(url) for p in exclude_compiled):
                continue

            # Mark as crawled
            self.crawled_urls.add(url)

            # Analyze URL
            result = await self.analyze_url(url, crawl_options)
            self.crawl_results[url] = result

            # Extract links and add to queue
            if result.get("success", False) and result.get("links", {}).get(
                "internal", []
            ):
                for link in result["links"]["internal"]:
                    link_url = link.get("url")
                    link_domain = urlparse(link_url).netloc

                    # Only follow links on the same domain
                    if (
                        link_url
                        and link_domain == start_domain
                        and link_url not in self.crawled_urls
                    ):
                        self.crawl_queue.append(link_url)

        return {
            "success": True,
            "start_url": start_url,
            "crawled_urls": list(self.crawled_urls),
            "page_count": len(self.crawled_urls),
            "session_id": self.session_id,
            "results": self.crawl_results,
        }

    async def close(self) -> None:
        """Close all resources."""
        await self.dynamic_processor.close()

    async def take_screenshot(
        self, url: str, output_path: Optional[str] = None, full_page: bool = True
    ) -> Optional[bytes]:
        """
        Take a screenshot of a webpage.

        Args:
            url: URL to screenshot
            output_path: Path to save screenshot (if None, returns bytes)
            full_page: Whether to capture full page or just viewport

        Returns:
            Screenshot as bytes if output_path is None, otherwise None
        """
        return await self.dynamic_processor.take_screenshot(url, output_path, full_page)

    def clear_cache(self, url: Optional[str] = None) -> None:
        """
        Clear the content cache.

        Args:
            url: Optional URL to clear (if None, clear all)
        """
        if url:
            asyncio.run(self.content_cache.delete(url))
        else:
            asyncio.run(self.content_cache.clear())

    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the analyzer."""
        return {
            "session_id": self.session_id,
            "processed_urls": len(self.processed_urls),
            "cache_stats": self.content_cache.stats,
        }
