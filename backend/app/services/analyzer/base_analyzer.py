from typing import Dict, Any, Optional


class BaseAnalyzer:
    """Base class for all content analyzers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration options."""
        self.config = config or {}

    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Base analyze method to be implemented by subclasses.

        Returns:
            Dict with analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze method")
