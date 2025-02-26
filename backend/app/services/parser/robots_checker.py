import asyncio
from typing import Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin
import re
import time

import aiohttp
from loguru import logger


class RobotsChecker:
    """
    Utility for parsing and checking robots.txt rules.
    """

    def __init__(self):
        """Initialize the robots checker."""
        # Cache robots.txt content
        self.robots_cache = {}  # domain -> (content, timestamp)
        self.cache_ttl = 3600  # 1 hour TTL for robots cache

        # Cache allowed/disallowed paths
        self.allowed_cache = {}  # (domain, user_agent) -> {allowed_paths}
        self.disallowed_cache = {}  # (domain, user_agent) -> {disallowed_paths}

        # Cache for URL checks
        self.check_cache = {}  # (url, user_agent) -> (is_allowed, timestamp)
        self.check_cache_ttl = 600  # 10 minutes TTL for check cache

    async def check_url(self, url: str, user_agent: str) -> bool:
        """
        Check if a URL is allowed by robots.txt.

        Args:
            url: URL to check
            user_agent: User agent to check against

        Returns:
            Boolean indicating if URL is allowed
        """
        # If no URL, allow
        if not url:
            return True

        # Check cache first
        cache_key = (url, user_agent)
        if cache_key in self.check_cache:
            result, timestamp = self.check_cache[cache_key]
            if time.time() - timestamp < self.check_cache_ttl:
                return result

        # Parse URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path

        if not domain or not path:
            return True

        # Default to allowed if robots.txt cannot be fetched
        is_allowed = True

        try:
            # Get robots.txt rules
            allowed_paths, disallowed_paths = await self._get_rules(domain, user_agent)

            # Check if URL matches disallowed paths
            is_allowed = not self._is_path_disallowed(
                path, disallowed_paths, allowed_paths
            )

            # Cache result
            self.check_cache[cache_key] = (is_allowed, time.time())

            return is_allowed

        except Exception as e:
            logger.warning(f"Error checking robots.txt for {domain}: {str(e)}")
            return True  # Default to allowed

    async def _get_rules(
        self, domain: str, user_agent: str
    ) -> Tuple[Set[str], Set[str]]:
        """
        Get allow/disallow rules for a domain and user agent.

        Args:
            domain: Domain to get rules for
            user_agent: User agent to match

        Returns:
            Tuple of (allowed_paths, disallowed_paths)
        """
        # Check cache first
        cache_key = (domain, user_agent)

        if cache_key in self.allowed_cache and cache_key in self.disallowed_cache:
            return self.allowed_cache[cache_key], self.disallowed_cache[cache_key]

        # Fetch and parse robots.txt
        robots_content = await self._fetch_robots_txt(domain)

        if robots_content is None:
            # No robots.txt found, everything is allowed
            self.allowed_cache[cache_key] = set()
            self.disallowed_cache[cache_key] = set()
            return set(), set()

        # Parse the rules
        allowed_paths, disallowed_paths = self._parse_robots_txt(
            robots_content, user_agent
        )

        # Cache the results
        self.allowed_cache[cache_key] = allowed_paths
        self.disallowed_cache[cache_key] = disallowed_paths

        return allowed_paths, disallowed_paths

    async def _fetch_robots_txt(self, domain: str) -> Optional[str]:
        """
        Fetch robots.txt content for a domain.

        Args:
            domain: Domain to fetch robots.txt for

        Returns:
            robots.txt content as string or None if not found
        """
        # Check cache first
        if domain in self.robots_cache:
            content, timestamp = self.robots_cache[domain]
            if time.time() - timestamp < self.cache_ttl:
                return content

        # Fetch robots.txt
        robots_url = f"https://{domain}/robots.txt"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        self.robots_cache[domain] = (content, time.time())
                        return content
                    elif response.status == 404:
                        # No robots.txt found, cache empty string
                        self.robots_cache[domain] = ("", time.time())
                        return ""
                    else:
                        # Try HTTP if HTTPS fails
                        robots_url = f"http://{domain}/robots.txt"
                        try:
                            async with session.get(
                                robots_url, timeout=10
                            ) as http_response:
                                if http_response.status == 200:
                                    content = await http_response.text()
                                    self.robots_cache[domain] = (content, time.time())
                                    return content
                                else:
                                    # Could not get robots.txt, cache empty string
                                    self.robots_cache[domain] = ("", time.time())
                                    return ""
                        except:
                            self.robots_cache[domain] = ("", time.time())
                            return ""
        except Exception as e:
            logger.debug(f"Error fetching robots.txt for {domain}: {str(e)}")
            # Cache empty string on error
            self.robots_cache[domain] = ("", time.time())
            return ""

    def _parse_robots_txt(
        self, content: str, user_agent: str
    ) -> Tuple[Set[str], Set[str]]:
        """
        Parse robots.txt content and extract rules for the user agent.

        Args:
            content: robots.txt content
            user_agent: User agent to match

        Returns:
            Tuple of (allowed_paths, disallowed_paths)
        """
        allowed_paths = set()
        disallowed_paths = set()

        # Normalize user agent to lowercase
        user_agent_lower = user_agent.lower()

        # Handle empty content
        if not content.strip():
            return allowed_paths, disallowed_paths

        # Split content into lines and clean
        lines = [line.strip() for line in content.lower().split("\n")]

        # Look for sections relevant to our user agent
        current_section_applies = False
        specific_agent_rules_found = False

        # Store rules for specific user agent and wildcard (*) user agent
        specific_agent_allowed = set()
        specific_agent_disallowed = set()
        wildcard_agent_allowed = set()
        wildcard_agent_disallowed = set()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Handle user-agent lines
            if line.startswith("user-agent:"):
                # If we've already found specific rules and this is a new user-agent,
                # stop processing
                if specific_agent_rules_found and not current_section_applies:
                    break

                agent = line[11:].strip()

                # Check if this section applies to our user agent
                if self._agent_matches(user_agent_lower, agent):
                    current_section_applies = True
                    # Mark if we found specific rules
                    if agent != "*":
                        specific_agent_rules_found = True
                else:
                    current_section_applies = False

            # Handle allow rules
            elif line.startswith("allow:"):
                path = line[6:].strip()
                if path:
                    if current_section_applies:
                        if specific_agent_rules_found:
                            specific_agent_allowed.add(path)
                        else:
                            wildcard_agent_allowed.add(path)

            # Handle disallow rules
            elif line.startswith("disallow:"):
                path = line[9:].strip()
                # Empty disallow means allow all
                if path:
                    if current_section_applies:
                        if specific_agent_rules_found:
                            specific_agent_disallowed.add(path)
                        else:
                            wildcard_agent_disallowed.add(path)

        # Use specific agent rules if found, otherwise use wildcard rules
        if specific_agent_rules_found:
            allowed_paths = specific_agent_allowed
            disallowed_paths = specific_agent_disallowed
        else:
            allowed_paths = wildcard_agent_allowed
            disallowed_paths = wildcard_agent_disallowed

        return allowed_paths, disallowed_paths

    def _agent_matches(self, user_agent: str, pattern: str) -> bool:
        """
        Check if a user agent matches a pattern.

        Args:
            user_agent: User agent to check
            pattern: Pattern to match against

        Returns:
            Boolean indicating if pattern matches
        """
        pattern = pattern.lower()

        # Handle wildcard pattern
        if pattern == "*":
            return True

        # Check if user agent contains the pattern
        return pattern in user_agent

    def _is_path_disallowed(
        self, path: str, disallowed_paths: Set[str], allowed_paths: Set[str]
    ) -> bool:
        """
        Check if a path is disallowed by robots.txt rules.

        Args:
            path: URL path to check
            disallowed_paths: Set of disallowed paths
            allowed_paths: Set of allowed paths (overrides disallowed paths)

        Returns:
            Boolean indicating if path is disallowed
        """
        # If no paths defined, everything is allowed
        if not disallowed_paths:
            return False

        # Check allow rules first (they take precedence)
        for allowed in allowed_paths:
            if self._path_matches(path, allowed):
                return False

        # Then check disallow rules
        for disallowed in disallowed_paths:
            if self._path_matches(path, disallowed):
                return True

        # Default to allowed
        return False

    def _path_matches(self, path: str, pattern: str) -> bool:
        """
        Check if a path matches a pattern.

        Args:
            path: URL path to check
            pattern: Pattern to match against

        Returns:
            Boolean indicating if pattern matches
        """
        # Ensure path starts with /
        if not path.startswith("/"):
            path = "/" + path

        # Exact match
        if pattern == path:
            return True

        # Handle wildcards (*)
        if "*" in pattern:
            pattern_regex = "^" + re.escape(pattern).replace("\\*", ".*") + "$"
            return bool(re.match(pattern_regex, path))

        # Handle path prefixes
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return path.startswith(prefix)

        # Handle directory matches
        if pattern.endswith("/"):
            return path == pattern or path.startswith(pattern)

        # Default path matching
        return path.startswith(pattern)

    def clear_cache(self, domain: Optional[str] = None) -> None:
        """
        Clear cached robots.txt data.

        Args:
            domain: Optional domain to clear cache for (if None, clear all)
        """
        if domain:
            # Clear specific domain
            if domain in self.robots_cache:
                del self.robots_cache[domain]

            # Clear rules for domain
            keys_to_remove = []
            for key in self.allowed_cache:
                if key[0] == domain:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                if key in self.allowed_cache:
                    del self.allowed_cache[key]
                if key in self.disallowed_cache:
                    del self.disallowed_cache[key]

            # Clear URL checks for domain
            check_keys_to_remove = []
            for key in self.check_cache:
                if urlparse(key[0]).netloc == domain:
                    check_keys_to_remove.append(key)

            for key in check_keys_to_remove:
                if key in self.check_cache:
                    del self.check_cache[key]
        else:
            # Clear all caches
            self.robots_cache = {}
            self.allowed_cache = {}
            self.disallowed_cache = {}
            self.check_cache = {}
