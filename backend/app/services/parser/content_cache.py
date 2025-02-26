import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional
import gzip
from io import BytesIO

from loguru import logger

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class ContentCache:
    """
    Cache for storing parsed content data.
    """

    def __init__(self, use_redis: bool = False, redis_url: Optional[str] = None):
        """
        Initialize the content cache.

        Args:
            use_redis: Whether to use Redis for caching
            redis_url: Redis connection URL (if using Redis)
        """
        self.memory_cache = {}  # url -> (data, timestamp, ttl)
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.redis_client = None
        self.redis_prefix = "webcontent_analyzer:cache:"

        # Initialize Redis if requested
        if self.use_redis:
            self._init_redis(redis_url)

    def _init_redis(self, redis_url: Optional[str]) -> None:
        """
        Initialize Redis connection.

        Args:
            redis_url: Redis connection URL
        """
        try:
            if redis_url:
                self.redis_client = redis.from_url(redis_url)
            else:
                self.redis_client = redis.Redis()
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            self.use_redis = False
            self.redis_client = None

    async def get(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached content for a URL.

        Args:
            url: URL to get cached content for

        Returns:
            Cached data or None if not in cache or expired
        """
        # Generate cache key
        cache_key = self._create_cache_key(url)

        # Try Redis first if enabled
        if self.use_redis and self.redis_client:
            data = await self._get_from_redis(cache_key)
            if data:
                return data

        # Fall back to memory cache
        return self._get_from_memory(cache_key)

    async def set(self, url: str, data: Dict[str, Any], ttl: int = 3600) -> None:
        """
        Store data in cache.

        Args:
            url: URL to cache data for
            data: Data to cache
            ttl: Time-to-live in seconds
        """
        # Generate cache key
        cache_key = self._create_cache_key(url)

        # Store in memory
        self._set_in_memory(cache_key, data, ttl)

        # Store in Redis if enabled
        if self.use_redis and self.redis_client:
            await self._set_in_redis(cache_key, data, ttl)

    async def delete(self, url: str) -> None:
        """
        Delete data from cache.

        Args:
            url: URL to delete from cache
        """
        cache_key = self._create_cache_key(url)

        # Remove from memory
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]

        # Remove from Redis if enabled
        if self.use_redis and self.redis_client:
            try:
                await self.redis_client.delete(self.redis_prefix + cache_key)
            except Exception as e:
                logger.error(f"Failed to delete from Redis: {str(e)}")

    async def clear(self) -> None:
        """Clear all cached data."""
        # Clear memory cache
        self.memory_cache = {}

        # Clear Redis cache if enabled
        if self.use_redis and self.redis_client:
            try:
                # Find all keys with our prefix
                pattern = f"{self.redis_prefix}*"
                keys = await self.redis_client.keys(pattern)

                # Delete keys in batches
                if keys:
                    for i in range(0, len(keys), 1000):
                        batch = keys[i : i + 1000]
                        await self.redis_client.delete(*batch)
            except Exception as e:
                logger.error(f"Failed to clear Redis cache: {str(e)}")

    def _create_cache_key(self, url: str) -> str:
        """
        Create a cache key for a URL.

        Args:
            url: URL to create key for

        Returns:
            Cache key string
        """
        # Create a deterministic, safe key based on URL
        return hashlib.md5(url.encode()).hexdigest()

    def _get_from_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from memory cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        if key in self.memory_cache:
            data, timestamp, ttl = self.memory_cache[key]

            # Check if data is expired
            if time.time() - timestamp <= ttl:
                return data
            else:
                # Remove expired data
                del self.memory_cache[key]

        return None

    def _set_in_memory(self, key: str, data: Dict[str, Any], ttl: int) -> None:
        """
        Store data in memory cache.

        Args:
            key: Cache key
            data: Data to store
            ttl: Time-to-live in seconds
        """
        # Store data with timestamp
        self.memory_cache[key] = (data, time.time(), ttl)

        # Clean up old entries if cache is too large (> 1000 items)
        if len(self.memory_cache) > 1000:
            self._cleanup_memory_cache()

    def _cleanup_memory_cache(self) -> None:
        """Clean up expired or old entries from memory cache."""
        now = time.time()

        # Remove expired entries
        expired_keys = [
            key
            for key, (_, timestamp, ttl) in self.memory_cache.items()
            if now - timestamp > ttl
        ]

        for key in expired_keys:
            del self.memory_cache[key]

        # If still too large, remove oldest entries
        if len(self.memory_cache) > 1000:
            # Sort by timestamp (oldest first)
            sorted_items = sorted(
                self.memory_cache.items(), key=lambda x: x[1][1]  # Sort by timestamp
            )

            # Keep only the 800 newest entries
            self.memory_cache = dict(sorted_items[-800:])

    async def _get_from_redis(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from Redis cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        if not self.redis_client:
            return None

        try:
            # Get data from Redis
            redis_key = self.redis_prefix + key
            compressed_data = await self.redis_client.get(redis_key)

            if compressed_data:
                # Decompress and deserialize data
                return self._deserialize_data(compressed_data)
        except Exception as e:
            logger.error(f"Failed to get data from Redis: {str(e)}")

        return None

    async def _set_in_redis(self, key: str, data: Dict[str, Any], ttl: int) -> None:
        """
        Store data in Redis cache.

        Args:
            key: Cache key
            data: Data to store
            ttl: Time-to-live in seconds
        """
        if not self.redis_client:
            return

        try:
            # Serialize and compress data
            compressed_data = self._serialize_data(data)

            # Store in Redis with TTL
            redis_key = self.redis_prefix + key
            await self.redis_client.setex(redis_key, ttl, compressed_data)
        except Exception as e:
            logger.error(f"Failed to store data in Redis: {str(e)}")

    def _serialize_data(self, data: Dict[str, Any]) -> bytes:
        """
        Serialize and compress data for storage.

        Args:
            data: Data to serialize

        Returns:
            Compressed serialized data
        """
        # Handle HTML content separately to save space
        optimized_data = data.copy()
        html_content = None

        if "html_content" in optimized_data and isinstance(
            optimized_data["html_content"], str
        ):
            html_content = optimized_data["html_content"]
            # Replace with placeholder
            optimized_data["html_content"] = "__COMPRESSED_HTML__"

        # Serialize to JSON
        json_data = json.dumps(optimized_data)

        # Compress using gzip
        buffer = BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
            f.write(json_data.encode("utf-8"))

            # Add HTML content at the end if it exists
            if html_content:
                f.write(b"\n__HTML_SEPARATOR__\n")
                f.write(html_content.encode("utf-8"))

        compressed_data = buffer.getvalue()

        return compressed_data

    def _deserialize_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """
        Decompress and deserialize data.

        Args:
            compressed_data: Compressed serialized data

        Returns:
            Original data
        """
        # Decompress data
        buffer = BytesIO(compressed_data)
        html_content = None

        with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
            decompressed_data = f.read().decode("utf-8")

            # Check if there's a separator
            if "\n__HTML_SEPARATOR__\n" in decompressed_data:
                json_data, html_content = decompressed_data.split(
                    "\n__HTML_SEPARATOR__\n", 1
                )
            else:
                json_data = decompressed_data

        # Deserialize JSON
        data = json.loads(json_data)

        # Restore HTML content if it exists
        if html_content and data.get("html_content") == "__COMPRESSED_HTML__":
            data["html_content"] = html_content

        return data

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "redis_enabled": self.use_redis and self.redis_client is not None,
            "redis_cache_size": 0,
        }

        # Get Redis stats if enabled
        if self.use_redis and self.redis_client:
            try:
                loop = asyncio.get_event_loop()
                stats["redis_cache_size"] = loop.run_until_complete(
                    self.redis_client.keys(f"{self.redis_prefix}*")
                )
            except Exception:
                pass

        return stats
