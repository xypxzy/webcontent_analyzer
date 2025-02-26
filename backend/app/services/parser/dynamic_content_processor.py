import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple
import time
import os
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from loguru import logger

from app.core.config import settings


class DynamicContentProcessor:
    """
    Handler for processing dynamic content using headless browsers.
    """

    def __init__(self):
        """Initialize the dynamic content processor."""
        self.browser = None
        self.context = None
        self.browser_lock = asyncio.Lock()
        self.max_wait_time = settings.PARSER_TIMEOUT
        self.browser_type = os.environ.get(
            "HEADLESS_BROWSER_TYPE", "chromium"
        )  # chromium, firefox, webkit
        self.resource_exclusions = [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.svg",
            "*.woff",
            "*.woff2",
            "*.ttf",
            "*.eot",
            "*.mp4",
            "*.webm",
            "*.mp3",
            "*.ogg",
            "*.pdf",
            "*.zip",
            "*.tar",
            "*.gz",
            "analytics",
            "googletagmanager",
            "facebook",
            "twitter",
            "linkedin",
            "doubleclick",
            "adsense",
        ]

    async def __aenter__(self):
        """Context manager enter."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        await self.close()

    async def _initialize_browser(self):
        """Initialize the browser if it hasn't been done yet."""
        if self.browser is None:
            async with self.browser_lock:
                if self.browser is None:  # Double check under lock
                    try:
                        playwright = await async_playwright().start()

                        # Choose browser based on configuration
                        if self.browser_type == "firefox":
                            browser_factory = playwright.firefox
                        elif self.browser_type == "webkit":
                            browser_factory = playwright.webkit
                        else:
                            browser_factory = playwright.chromium

                        # Launch browser with appropriate settings
                        self.browser = await browser_factory.launch(
                            headless=True,
                            args=[
                                "--disable-gpu",
                                "--disable-dev-shm-usage",
                                "--disable-setuid-sandbox",
                                "--no-sandbox",
                                "--disable-features=IsolateOrigins,site-per-process",
                            ],
                        )

                        # Create context with common settings
                        self.context = await self.browser.new_context(
                            viewport={"width": 1920, "height": 1080},
                            user_agent=settings.PARSER_USER_AGENT,
                        )

                        logger.info(
                            f"Initialized {self.browser_type} browser for dynamic content processing"
                        )
                    except Exception as e:
                        logger.error(f"Failed to initialize browser: {str(e)}")
                        raise

    async def close(self):
        """Close the browser and free resources."""
        if self.browser:
            try:
                await self.browser.close()
                logger.info("Closed headless browser")
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")
            finally:
                self.browser = None
                self.context = None

    async def render_page(
        self,
        url: str,
        user_interactions: List[Dict] = None,
        wait_for_selector: str = None,
        wait_until: str = "networkidle",
        timeout: int = None,
    ) -> str:
        """
        Fetch URL content with a headless browser to render JavaScript content.

        Args:
            url: URL to render
            user_interactions: List of user interactions to perform
            wait_for_selector: CSS selector to wait for before considering page loaded
            wait_until: Event to wait for (networkidle, load, domcontentloaded)
            timeout: Custom timeout in seconds

        Returns:
            Rendered HTML content
        """
        # Set default values
        if user_interactions is None:
            user_interactions = []

        if timeout is None:
            timeout = self.max_wait_time

        # Initialize browser if needed
        await self._initialize_browser()

        try:
            # Create a new page
            page = await self.context.new_page()

            # Block unnecessary resources to speed up loading
            await self._setup_page_optimizations(page)

            # Navigate to URL with appropriate wait conditions
            response = await page.goto(
                url, wait_until=wait_until, timeout=timeout * 1000
            )

            # Check response status
            if not response or response.status >= 400:
                status = response.status if response else "Unknown"
                logger.warning(f"Failed to load URL: {url}, status: {status}")
                await page.close()
                return None

            # Wait for specific selector if provided
            if wait_for_selector:
                try:
                    await page.wait_for_selector(
                        wait_for_selector, timeout=timeout * 1000
                    )
                except Exception as e:
                    logger.warning(f"Selector {wait_for_selector} not found: {str(e)}")

            # Ensure page is fully loaded
            if wait_until == "networkidle":
                # Additional wait for any lingering AJAX requests
                await self._wait_for_network_idle(page, timeout=5)

            # Perform any requested user interactions
            await self._perform_user_interactions(page, user_interactions)

            # Wait for a moment after interactions
            await asyncio.sleep(1)

            # Get final HTML content
            html_content = await page.content()

            # Close the page
            await page.close()

            return html_content

        except Exception as e:
            logger.error(f"Error rendering page {url}: {str(e)}")
            return None

    async def _setup_page_optimizations(self, page: Page):
        """
        Set up page optimizations to speed up loading.

        Args:
            page: Playwright Page object
        """
        # Block unnecessary resources
        await page.route(
            re.compile("|".join(self.resource_exclusions).replace(".", "\.")),
            lambda route: route.abort(),
        )

        # Intercept and cache requests
        await page.route(
            "**/*",
            lambda route: (
                route.continue_()
                if self._should_process_request(route.request.url)
                else route.abort()
            ),
        )

    def _should_process_request(self, url: str) -> bool:
        """
        Determine if a request should be processed or blocked.

        Args:
            url: Request URL

        Returns:
            Boolean indicating if request should be processed
        """
        # Skip tracking scripts and unnecessary resources
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        if any(
            tracker in domain for tracker in ["analytics", "tracking", "pixel", "tag"]
        ):
            return False

        path = parsed_url.path.lower()
        ext = os.path.splitext(path)[1]

        # Process HTML, CSS, JS, JSON
        if ext in [".html", ".htm", ".css", ".js", ".json", ""]:
            return True

        return False

    async def _wait_for_network_idle(self, page: Page, timeout: int = 5):
        """
        Wait for network to be idle (no requests for a period).

        Args:
            page: Playwright Page object
            timeout: Time to wait in seconds
        """
        try:
            # Check if there are any pending network requests
            await page.wait_for_function(
                """() => {
                    return window.performance.getEntriesByType('resource')
                        .filter(r => !r.responseEnd)
                        .length === 0;
                }""",
                timeout=timeout * 1000,
            )
        except Exception as e:
            # Timeout is acceptable here
            logger.debug(f"Network idle wait timed out: {str(e)}")

    async def _perform_user_interactions(self, page: Page, interactions: List[Dict]):
        """
        Perform user interactions on the page.

        Args:
            page: Playwright Page object
            interactions: List of interaction objects
        """
        for interaction in interactions:
            interaction_type = interaction.get("type", "").lower()

            try:
                if interaction_type == "click":
                    # Click on an element
                    selector = interaction.get("selector")
                    if selector:
                        await page.click(selector)
                        await asyncio.sleep(0.5)  # Small delay after click

                elif interaction_type == "scroll":
                    # Scroll the page
                    amount = interaction.get("amount", 300)
                    await page.evaluate(f"window.scrollBy(0, {amount})")
                    await asyncio.sleep(0.5)  # Small delay after scroll

                elif interaction_type == "wait":
                    # Wait for a specified time
                    wait_time = min(interaction.get("time", 1), 10)  # Cap at 10 seconds
                    await asyncio.sleep(wait_time)

                elif interaction_type == "input":
                    # Enter text into an input field
                    selector = interaction.get("selector")
                    value = interaction.get("value", "")
                    if selector:
                        await page.fill(selector, value)

                elif interaction_type == "select":
                    # Select an option from a dropdown
                    selector = interaction.get("selector")
                    value = interaction.get("value")
                    if selector and value is not None:
                        await page.select_option(selector, value)

                elif interaction_type == "hover":
                    # Hover over an element
                    selector = interaction.get("selector")
                    if selector:
                        await page.hover(selector)

                elif interaction_type == "press":
                    # Press a keyboard key
                    key = interaction.get("key")
                    if key:
                        await page.keyboard.press(key)

                elif interaction_type == "submit":
                    # Submit a form
                    selector = interaction.get("selector", "form")
                    if selector:
                        await page.evaluate(
                            f"document.querySelector('{selector}').submit()"
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to perform interaction {interaction_type}: {str(e)}"
                )

    async def take_screenshot(
        self, url: str, output_path: str = None, full_page: bool = True
    ) -> Optional[bytes]:
        """
        Take a screenshot of a webpage.

        Args:
            url: URL to screenshot
            output_path: Path to save screenshot (if None, returns bytes)
            full_page: Whether to capture the full page or just viewport

        Returns:
            Screenshot bytes if output_path is None, otherwise None
        """
        # Initialize browser if needed
        await self._initialize_browser()

        try:
            # Create a new page
            page = await self.context.new_page()

            # Navigate to URL
            await page.goto(url, wait_until="networkidle")

            # Take screenshot
            if output_path:
                await page.screenshot(path=output_path, full_page=full_page)
                result = None
            else:
                result = await page.screenshot(full_page=full_page)

            # Close page
            await page.close()

            return result

        except Exception as e:
            logger.error(f"Error taking screenshot of {url}: {str(e)}")
            return None

    async def extract_ajax_requests(self, url: str) -> List[Dict[str, Any]]:
        """
        Extract AJAX requests made by the page.

        Args:
            url: URL to analyze

        Returns:
            List of request information dictionaries
        """
        # Initialize browser if needed
        await self._initialize_browser()

        ajax_requests = []

        try:
            # Create a new page
            page = await self.context.new_page()

            # Set up request interception
            await page.route("**/*", lambda route: route.continue_())

            # Listen for requests
            page.on(
                "request", lambda request: self._capture_request(request, ajax_requests)
            )

            # Navigate to URL
            await page.goto(url, wait_until="networkidle")

            # Wait for any additional requests
            await asyncio.sleep(5)

            # Close page
            await page.close()

            return ajax_requests

        except Exception as e:
            logger.error(f"Error extracting AJAX requests from {url}: {str(e)}")
            return []

    def _capture_request(self, request, requests_list):
        """
        Capture request information for AJAX analysis.

        Args:
            request: Playwright Request object
            requests_list: List to append request info to
        """
        # Skip non-AJAX requests
        if not self._is_ajax_request(request):
            return

        # Capture request details
        request_info = {
            "url": request.url,
            "method": request.method,
            "headers": request.headers,
            "resource_type": request.resource_type,
            "timestamp": time.time(),
        }

        # Add to list
        requests_list.append(request_info)

    def _is_ajax_request(self, request):
        """
        Determine if a request is an AJAX request.

        Args:
            request: Playwright Request object

        Returns:
            Boolean indicating if request is AJAX
        """
        # Check for XHR or Fetch requests
        if request.resource_type in ["xhr", "fetch"]:
            return True

        # Check for JSON responses
        if "application/json" in request.headers.get("accept", ""):
            return True

        # Check for AJAX headers
        if "X-Requested-With" in request.headers:
            return True

        return False

    async def extract_lazy_loaded_content(
        self, url: str, scroll_steps: int = 5, wait_time: int = 2
    ) -> str:
        """
        Extract content that is lazy-loaded as user scrolls.

        Args:
            url: URL to analyze
            scroll_steps: Number of scroll steps to perform
            wait_time: Time to wait between scrolls

        Returns:
            HTML content after scrolling
        """
        # Initialize browser if needed
        await self._initialize_browser()

        try:
            # Create a new page
            page = await self.context.new_page()

            # Navigate to URL
            await page.goto(url, wait_until="networkidle")

            # Get initial height
            height = await page.evaluate("document.body.scrollHeight")

            # Scroll down in steps
            for step in range(scroll_steps):
                scroll_amount = height * (step + 1) / scroll_steps
                await page.evaluate(f"window.scrollTo(0, {scroll_amount})")
                await asyncio.sleep(wait_time)  # Wait for content to load

            # Get final HTML content
            html_content = await page.content()

            # Close page
            await page.close()

            return html_content

        except Exception as e:
            logger.error(f"Error extracting lazy-loaded content from {url}: {str(e)}")
            return None

    async def is_dynamic_page(self, url: str) -> Tuple[bool, str]:
        """
        Determine if a page is likely dynamic (requires JS rendering).

        Args:
            url: URL to analyze

        Returns:
            Tuple of (is_dynamic, reason)
        """
        # Initialize browser if needed
        await self._initialize_browser()

        try:
            # Create a new page
            page = await self.context.new_page()

            # Navigate to URL without JavaScript
            await page.set_javascript_enabled(False)
            response_without_js = await page.goto(url, wait_until="domcontentloaded")
            html_without_js = await page.content()

            # Navigate to the same URL with JavaScript enabled
            await page.set_javascript_enabled(True)
            await page.reload(wait_until="networkidle")
            html_with_js = await page.content()

            # Compare content length
            without_js_text = await page.evaluate("document.body.innerText")
            await page.reload(wait_until="networkidle")  # Reload with JS enabled
            with_js_text = await page.evaluate("document.body.innerText")

            # Check for significant difference
            text_diff_ratio = len(with_js_text) / max(len(without_js_text), 1)
            html_diff_ratio = len(html_with_js) / max(len(html_without_js), 1)

            # Close page
            await page.close()

            # Determine if dynamic based on content difference
            is_dynamic = text_diff_ratio > 1.3 or html_diff_ratio > 1.5

            reason = ""
            if text_diff_ratio > 1.3:
                reason = f"Visible text content differs significantly with JS enabled (ratio: {text_diff_ratio:.2f})"
            elif html_diff_ratio > 1.5:
                reason = f"HTML structure differs significantly with JS enabled (ratio: {html_diff_ratio:.2f})"

            return is_dynamic, reason

        except Exception as e:
            logger.error(f"Error checking if {url} is dynamic: {str(e)}")
            return False, f"Error checking: {str(e)}"
