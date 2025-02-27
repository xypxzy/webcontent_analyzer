import json
import asyncio
import pytest
from bs4 import BeautifulSoup

from app.services.analyzer.seo_analyzer import SEOAnalyzer
from app.services.analyzer.content_analyzer import ContentAnalyzer


@pytest.mark.asyncio
async def test_seo_analyzer():
    """Test the SEO analyzer with a sample HTML page."""
    # Load test HTML
    with open("app/tests/fixtures/sample_page.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Create analyzer
    analyzer = SEOAnalyzer()

    # Extract text and metadata
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    metadata = {
        "title": soup.title.string if soup.title else "",
        "description": (
            soup.find("meta", attrs={"name": "description"}).get("content", "")
            if soup.find("meta", attrs={"name": "description"})
            else ""
        ),
        "keywords": (
            [
                k.strip()
                for k in soup.find("meta", attrs={"name": "keywords"})
                .get("content", "")
                .split(",")
            ]
            if soup.find("meta", attrs={"name": "keywords"})
            else []
        ),
    }

    # Define target keywords
    target_keywords = ["web content", "analysis", "SEO"]

    # Run analysis
    result = await analyzer.analyze(
        html=html,
        text=text,
        metadata=metadata,
        url="https://example.com/test-page",
        target_keywords=target_keywords,
    )

    # Basic assertions
    assert isinstance(result, dict)
    assert "meta_tags" in result
    assert "headings" in result
    assert "url_structure" in result
    assert "keyword_usage" in result
    assert "overall_score" in result
    assert "recommendations" in result

    # Print sample output for debugging
    print(json.dumps(result, indent=2))

    # Verify score is in valid range
    assert 0 <= result["overall_score"] <= 1.0


@pytest.mark.asyncio
async def test_content_analyzer_integration():
    """Test the integration of SEO analyzer with the main content analyzer."""
    # Load test HTML
    with open("app/tests/fixtures/sample_page.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Create main analyzer
    analyzer = ContentAnalyzer()

    # Extract text and metadata
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    metadata = {
        "title": soup.title.string if soup.title else "",
        "description": (
            soup.find("meta", attrs={"name": "description"}).get("content", "")
            if soup.find("meta", attrs={"name": "description"})
            else ""
        ),
        "keywords": (
            [
                k.strip()
                for k in soup.find("meta", attrs={"name": "keywords"})
                .get("content", "")
                .split(",")
            ]
            if soup.find("meta", attrs={"name": "keywords"})
            else []
        ),
    }

    # Run analysis
    result = await analyzer.analyze_content(
        text=text, html=html, metadata=metadata, url="https://example.com/test-page"
    )

    # Basic assertions
    assert isinstance(result, dict)
    assert "seo_metrics" in result
    assert "recommendations" in result
    assert "analyzed_at" in result

    # Print sample output for debugging
    print(json.dumps(result, indent=2))
