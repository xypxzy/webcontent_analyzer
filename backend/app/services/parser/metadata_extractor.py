from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import json
import re


def extract_metadata(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """
    Extract metadata from HTML document.

    Args:
        soup: BeautifulSoup object
        url: URL of the page

    Returns:
        Dictionary containing metadata
    """
    metadata = {
        # Basic metadata
        "title": _extract_title(soup),
        "description": _extract_description(soup),
        "keywords": _extract_keywords(soup),
        "author": _extract_author(soup),
        "canonical_url": _extract_canonical(soup, url),
        "language": _extract_language(soup),
        # Technical metadata
        "charset": _extract_charset(soup),
        "viewport": _extract_viewport(soup),
        "robots": _extract_robots(soup),
        # Social media metadata
        "open_graph": _extract_opengraph(soup),
        "twitter_cards": _extract_twitter_cards(soup),
        # Structured data
        "json_ld": _extract_json_ld(soup),
        "microdata": extract_microdata(soup),
        # Link metadata
        "rel_links": _extract_rel_links(soup, url),
        "hreflang": _extract_hreflang(soup),
        # Additional metadata
        "icons": _extract_icons(soup, url),
        "page_type": _determine_page_type(soup),
    }

    return metadata


def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract page title with fallbacks.

    Args:
        soup: BeautifulSoup object

    Returns:
        Page title or None
    """
    # Check meta title first (OpenGraph)
    og_title = soup.find("meta", property="og:title")
    if og_title and "content" in og_title.attrs:
        return og_title["content"].strip()

    # Check title tag
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        return title_tag.string.strip()

    # Fallback to h1
    h1_tag = soup.find("h1")
    if h1_tag:
        return h1_tag.get_text().strip()

    return None


def _extract_description(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract page description with fallbacks.

    Args:
        soup: BeautifulSoup object

    Returns:
        Page description or None
    """
    # Check meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and "content" in meta_desc.attrs:
        return meta_desc["content"].strip()

    # Check OpenGraph description
    og_desc = soup.find("meta", property="og:description")
    if og_desc and "content" in og_desc.attrs:
        return og_desc["content"].strip()

    # Check Twitter description
    twitter_desc = soup.find("meta", attrs={"name": "twitter:description"})
    if twitter_desc and "content" in twitter_desc.attrs:
        return twitter_desc["content"].strip()

    return None


def _extract_keywords(soup: BeautifulSoup) -> List[str]:
    """
    Extract keywords from meta tags.

    Args:
        soup: BeautifulSoup object

    Returns:
        List of keywords
    """
    keywords = []

    # Check meta keywords
    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    if meta_keywords and "content" in meta_keywords.attrs:
        content = meta_keywords["content"].strip()
        if content:
            keywords = [k.strip() for k in content.split(",") if k.strip()]

    # Check article:tag metadata (common in blogs)
    article_tags = soup.find_all("meta", property="article:tag")
    for tag in article_tags:
        if "content" in tag.attrs and tag["content"].strip():
            keywords.append(tag["content"].strip())

    return keywords


def _extract_author(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract author information.

    Args:
        soup: BeautifulSoup object

    Returns:
        Author name or None
    """
    # Check meta author
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and "content" in meta_author.attrs:
        return meta_author["content"].strip()

    # Check OpenGraph article author
    og_author = soup.find("meta", property="article:author")
    if og_author and "content" in og_author.attrs:
        return og_author["content"].strip()

    # Check for author in RelativeRDFa
    author_rel = soup.find(rel="author")
    if author_rel:
        return author_rel.get_text().strip()

    # Check for schema.org author
    author_prop = soup.find(itemprop="author")
    if author_prop:
        name_prop = author_prop.find(itemprop="name")
        if name_prop:
            return name_prop.get_text().strip()
        return author_prop.get_text().strip()

    return None


def _extract_canonical(soup: BeautifulSoup, url: str) -> Optional[str]:
    """
    Extract canonical URL.

    Args:
        soup: BeautifulSoup object
        url: Original URL

    Returns:
        Canonical URL or None
    """
    # Check link rel=canonical
    canonical_link = soup.find("link", rel="canonical")
    if canonical_link and "href" in canonical_link.attrs:
        return urljoin(url, canonical_link["href"])

    # Check Open Graph URL
    og_url = soup.find("meta", property="og:url")
    if og_url and "content" in og_url.attrs:
        return og_url["content"]

    return url


def _extract_language(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract document language.

    Args:
        soup: BeautifulSoup object

    Returns:
        Language code or None
    """
    html_tag = soup.find("html")
    if html_tag and "lang" in html_tag.attrs:
        return html_tag["lang"]

    # Check for language meta tag
    meta_lang = soup.find("meta", attrs={"http-equiv": "content-language"})
    if meta_lang and "content" in meta_lang.attrs:
        return meta_lang["content"]

    # Check for language in Open Graph
    og_locale = soup.find("meta", property="og:locale")
    if og_locale and "content" in og_locale.attrs:
        return og_locale["content"]

    return None


def _extract_charset(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract document character set.

    Args:
        soup: BeautifulSoup object

    Returns:
        Character set or None
    """
    # Check meta charset
    meta_charset = soup.find("meta", charset=True)
    if meta_charset:
        return meta_charset["charset"]

    # Check http-equiv
    meta_content_type = soup.find("meta", attrs={"http-equiv": "Content-Type"})
    if meta_content_type and "content" in meta_content_type.attrs:
        match = re.search(r"charset=([^;]+)", meta_content_type["content"])
        if match:
            return match.group(1)

    return None


def _extract_viewport(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract viewport settings.

    Args:
        soup: BeautifulSoup object

    Returns:
        Viewport string or None
    """
    meta_viewport = soup.find("meta", attrs={"name": "viewport"})
    if meta_viewport and "content" in meta_viewport.attrs:
        return meta_viewport["content"]

    return None


def _extract_robots(soup: BeautifulSoup) -> Dict[str, bool]:
    """
    Extract robots directives.

    Args:
        soup: BeautifulSoup object

    Returns:
        Dictionary of robots directives
    """
    robots = {
        "index": True,
        "follow": True,
        "archive": True,
        "imageindex": True,
        "snippet": True,
        "translate": True,
    }

    meta_robots = soup.find("meta", attrs={"name": "robots"})
    if meta_robots and "content" in meta_robots.attrs:
        directives = [d.strip().lower() for d in meta_robots["content"].split(",")]

        # Process directives
        for directive in directives:
            if directive == "none":
                robots = {k: False for k in robots}
                break

            elif directive == "all":
                robots = {k: True for k in robots}
                break

            elif directive == "noindex":
                robots["index"] = False

            elif directive == "nofollow":
                robots["follow"] = False

            elif directive == "noarchive":
                robots["archive"] = False

            elif directive == "noimageindex":
                robots["imageindex"] = False

            elif directive == "nosnippet":
                robots["snippet"] = False

            elif directive == "notranslate":
                robots["translate"] = False

    return robots


def _extract_opengraph(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extract Open Graph metadata.

    Args:
        soup: BeautifulSoup object

    Returns:
        Dictionary of Open Graph metadata
    """
    og_meta = {}

    for meta in soup.find_all("meta", property=lambda x: x and x.startswith("og:")):
        if "content" in meta.attrs:
            property_name = meta["property"][3:]  # Remove 'og:' prefix
            og_meta[property_name] = meta["content"]

    return og_meta


def _extract_twitter_cards(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extract Twitter Card metadata.

    Args:
        soup: BeautifulSoup object

    Returns:
        Dictionary of Twitter Card metadata
    """
    twitter_meta = {}

    for meta in soup.find_all(
        "meta", attrs={"name": lambda x: x and x.startswith("twitter:")}
    ):
        if "content" in meta.attrs:
            property_name = meta["name"][8:]  # Remove 'twitter:' prefix
            twitter_meta[property_name] = meta["content"]

    return twitter_meta


def _extract_json_ld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Extract JSON-LD structured data.

    Args:
        soup: BeautifulSoup object

    Returns:
        List of JSON-LD objects
    """
    json_ld_data = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            json_ld_data.append(data)
        except (json.JSONDecodeError, TypeError):
            pass

    return json_ld_data


def extract_microdata(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Extract Microdata from HTML.

    Args:
        soup: BeautifulSoup object

    Returns:
        List of extracted Microdata items
    """
    items = []

    # Find all elements with itemscope attribute
    for element in soup.find_all(itemscope=True):
        item = {}

        # Get item type
        if element.has_attr("itemtype"):
            item["type"] = element["itemtype"]

        # Get item properties
        properties = {}
        for prop in element.find_all(itemprop=True):
            prop_name = prop["itemprop"]

            # Extract property value based on tag type
            if prop.name == "meta":
                prop_value = prop.get("content", "")
            elif prop.name == "img":
                prop_value = prop.get("src", "")
            elif prop.name == "a":
                prop_value = prop.get("href", "")
            elif prop.name == "time":
                prop_value = prop.get("datetime", prop.get_text())
            elif prop.name == "link":
                prop_value = prop.get("href", "")
            else:
                prop_value = prop.get_text().strip()

            # Handle nested itemscope
            if prop.has_attr("itemscope"):
                # Skip this property as it will be processed separately
                continue

            # Add to properties
            properties[prop_name] = prop_value

        item["properties"] = properties

        if properties:  # Only add if we found properties
            items.append(item)

    return items


def _extract_rel_links(soup: BeautifulSoup, url: str) -> Dict[str, str]:
    """
    Extract link elements with rel attribute.

    Args:
        soup: BeautifulSoup object
        url: Base URL

    Returns:
        Dictionary of rel links
    """
    rel_links = {}

    for link in soup.find_all("link", rel=True):
        if "href" in link.attrs:
            rel = link["rel"][0] if isinstance(link["rel"], list) else link["rel"]
            href = urljoin(url, link["href"])
            rel_links[rel] = href

    return rel_links


def _extract_hreflang(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extract hreflang links.

    Args:
        soup: BeautifulSoup object

    Returns:
        Dictionary of language to URL mappings
    """
    hreflang_links = {}

    for link in soup.find_all("link", rel="alternate"):
        if "hreflang" in link.attrs and "href" in link.attrs:
            hreflang_links[link["hreflang"]] = link["href"]

    return hreflang_links


def _extract_icons(soup: BeautifulSoup, url: str) -> Dict[str, str]:
    """
    Extract favicon and other icon links.

    Args:
        soup: BeautifulSoup object
        url: Base URL

    Returns:
        Dictionary of icon links
    """
    icons = {}

    # Apple touch icons
    apple_icon = soup.find("link", rel="apple-touch-icon")
    if apple_icon and "href" in apple_icon.attrs:
        icons["apple_touch_icon"] = urljoin(url, apple_icon["href"])

    # Standard favicon
    favicon = soup.find(
        "link", rel=lambda r: r and ("icon" in r or "shortcut icon" in r)
    )
    if favicon and "href" in favicon.attrs:
        icons["favicon"] = urljoin(url, favicon["href"])
    else:
        # Default favicon location
        icons["favicon"] = urljoin(url, "/favicon.ico")

    # Icon with multiple sizes
    for icon in soup.find_all("link", rel="icon"):
        if "sizes" in icon.attrs and "href" in icon.attrs:
            icons[f"icon_{icon['sizes']}"] = urljoin(url, icon["href"])

    return icons


def _determine_page_type(soup: BeautifulSoup) -> str:
    """
    Determine the type of page based on content.

    Args:
        soup: BeautifulSoup object

    Returns:
        Page type string
    """
    # Check OpenGraph type
    og_type = soup.find("meta", property="og:type")
    if og_type and "content" in og_type.attrs:
        return og_type["content"]

    # Look for common page patterns
    if soup.find("article"):
        return "article"

    if soup.find(itemtype=re.compile(r"Product|Offer")):
        return "product"

    if soup.find("form", attrs={"id": re.compile(r"checkout|cart", re.I)}):
        return "checkout"

    if soup.find("form", attrs={"id": re.compile(r"contact|feedback", re.I)}):
        return "contact"

    if soup.find(re.compile(r"login|signin", re.I)):
        return "login"

    # Check for blog indicators
    if soup.find("article") or soup.find(class_=re.compile(r"post|blog|article", re.I)):
        return "blog"

    # Look for listing page indicators
    if soup.find_all(class_=re.compile(r"product|listing|item-list", re.I)):
        return "listing"

    return "general"
