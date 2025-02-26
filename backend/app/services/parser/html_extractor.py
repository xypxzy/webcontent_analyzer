from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup, Tag
import re
import html2text


def extract_main_content(soup: BeautifulSoup) -> Tag:
    """
    Extract the main content from a web page, filtering out navigation, ads, etc.

    Args:
        soup: BeautifulSoup object of the HTML document

    Returns:
        BeautifulSoup Tag containing the main content
    """
    # Priority containers to check for main content
    main_content_selectors = [
        "main",
        "article",
        "div[role='main']",
        "#content",
        "#main",
        "#main-content",
        ".content",
        ".main",
        ".main-content",
        ".post",
        ".article",
        ".entry",
    ]

    # Try each selector in order of priority
    for selector in main_content_selectors:
        content = soup.select_one(selector)
        if content:
            # Check if it contains substantial text
            if len(content.get_text(strip=True)) > 100:
                return content

    # If no specialized containers found, try content extraction algorithm
    main_content = _extract_content_by_density(soup)
    if main_content:
        return main_content

    # Fallback: Create a new container with all potential content elements
    content_wrapper = soup.new_tag("div")
    content_wrapper["class"] = "extracted-content"

    # Try to get the body element
    body = soup.body or soup

    # Exclude common non-content elements
    non_content_tags = ["nav", "header", "footer", "aside", "style", "script", "iframe"]
    non_content_classes = [
        "nav",
        "menu",
        "header",
        "footer",
        "sidebar",
        "widget",
        "banner",
        "ad",
        "comment",
    ]
    non_content_ids = ["header", "footer", "sidebar", "menu", "nav", "comments"]

    # Add all potential content paragraphs, headings, and sections
    for elem in body.find_all(
        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "section"]
    ):
        # Skip if in non-content element
        if any(elem.find_parent(tag) for tag in non_content_tags):
            continue

        # Skip if it has a non-content class or id
        elem_classes = elem.get("class", [])
        elem_id = elem.get("id", "")

        if isinstance(elem_classes, str):
            elem_classes = [elem_classes]

        if any(
            cls.lower() in non_content_classes for cls in elem_classes if cls
        ) or any(nc_id in elem_id.lower() for nc_id in non_content_ids if elem_id):
            continue

        # Skip if too short (likely not main content)
        if len(elem.get_text(strip=True)) < 15 and elem.name not in [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        ]:
            continue

        # Clone the element and add to wrapper
        content_wrapper.append(elem.extract())

    # If we couldn't find substantial content, return the body
    if len(content_wrapper.get_text(strip=True)) < 200:
        return body

    return content_wrapper


def _extract_content_by_density(soup: BeautifulSoup) -> Optional[Tag]:
    """
    Extract content based on text density and tag distribution.

    Args:
        soup: BeautifulSoup object

    Returns:
        Most likely content container Tag or None
    """
    candidates = {}

    # Examine div, section and article tags as potential content containers
    for elem in soup.find_all(["div", "section", "article"]):
        # Skip very small elements
        if len(elem.get_text(strip=True)) < 100:
            continue

        # Skip elements with non-content IDs or classes
        elem_classes = " ".join(elem.get("class", []))
        elem_id = elem.get("id", "")

        if re.search(
            r"comment|sidebar|widget|banner|ad|promo|header|footer|nav|menu",
            f"{elem_classes} {elem_id}",
            re.I,
        ):
            continue

        # Calculate text density (text length / HTML length)
        text = elem.get_text(strip=True)
        html = str(elem)

        if len(html) == 0:
            continue

        text_density = len(text) / len(html)

        # Calculate link density (link text / total text)
        links_text = "".join([a.get_text(strip=True) for a in elem.find_all("a")])
        link_density = len(links_text) / len(text) if len(text) > 0 else 1

        # Calculate paragraph density (num paragraphs / container size)
        paragraphs = elem.find_all("p")
        p_density = len(paragraphs) * 1000 / len(html) if len(html) > 0 else 0

        # Calculate heading density
        headings = elem.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        heading_density = len(headings) * 1000 / len(html) if len(html) > 0 else 0

        # Calculate content score
        content_score = (
            text_density * 30  # Higher text density is good
            + (1 - link_density) * 20  # Lower link density is good
            + p_density * 15  # More paragraphs is good
            + heading_density * 10  # Some headings are good
            + len(text) / 1000  # Length bonus
        )

        # Apply penalties for non-content indicators
        if re.search(
            r"comment|sidebar|widget|banner|ad", elem_classes + " " + elem_id, re.I
        ):
            content_score *= 0.2

        # Store candidate with score
        candidates[elem] = content_score

    # Find best candidate
    if not candidates:
        return None

    best_candidate = max(candidates.items(), key=lambda x: x[1])

    # Only return if score is significant
    if best_candidate[1] > 20:
        return best_candidate[0]

    return None


def extract_text_content(html_content: Tag) -> str:
    """
    Extract clean text content from HTML.

    Args:
        html_content: HTML content as BeautifulSoup Tag

    Returns:
        Clean text content
    """
    # Create a copy to avoid modifying the original
    content_copy = BeautifulSoup(str(html_content), "html.parser")

    # Remove unwanted elements
    for elem in content_copy.find_all(
        ["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav"]
    ):
        elem.decompose()

    # Configure HTML to text converter
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    h2t.body_width = 0  # Don't wrap text
    h2t.ignore_emphasis = False
    h2t.ignore_images = True

    # Convert to markdown and then clean up
    text = h2t.handle(str(content_copy))

    # Clean up the text
    text = re.sub(r"\n{3,}", "\n\n", text)  # Remove excessive newlines
    text = re.sub(
        r"\[(.+?)\]\((.+?)\)", r"\1", text
    )  # Convert markdown links to just text
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # Remove images
    text = re.sub(r"</?[a-z][^>]*>", "", text)  # Remove any remaining HTML tags

    return text.strip()


def extract_metadata(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """
    Extract metadata from HTML document.

    Args:
        soup: BeautifulSoup object
        url: URL of the page

    Returns:
        Dictionary containing metadata
    """
    metadata = {}

    # Extract title
    title_tag = soup.find("title")
    if title_tag:
        metadata["title"] = title_tag.get_text().strip()
    else:
        # Try Open Graph title as fallback
        og_title = soup.find("meta", property="og:title")
        if og_title and "content" in og_title.attrs:
            metadata["title"] = og_title["content"].strip()

    # Extract description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and "content" in meta_desc.attrs:
        metadata["description"] = meta_desc["content"].strip()
    else:
        # Try Open Graph description as fallback
        og_desc = soup.find("meta", property="og:description")
        if og_desc and "content" in og_desc.attrs:
            metadata["description"] = og_desc["content"].strip()

    # Extract keywords
    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    if meta_keywords and "content" in meta_keywords.attrs:
        keywords = [k.strip() for k in meta_keywords["content"].split(",")]
        metadata["keywords"] = [k for k in keywords if k]

    # Extract canonical URL
    canonical = soup.find("link", rel="canonical")
    if canonical and "href" in canonical.attrs:
        metadata["canonical"] = canonical["href"]

    # Extract language
    html_tag = soup.find("html")
    if html_tag and "lang" in html_tag.attrs:
        metadata["language"] = html_tag["lang"]

    # Extract Open Graph metadata
    metadata["open_graph"] = extract_opengraph(soup)

    # Extract Twitter Card metadata
    metadata["twitter_card"] = {}
    for meta in soup.find_all("meta", attrs={"name": re.compile(r"^twitter:")}):
        if "content" in meta.attrs:
            property_name = meta["name"][8:]  # Remove 'twitter:' prefix
            metadata["twitter_card"][property_name] = meta["content"]

    # Extract author information
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and "content" in meta_author.attrs:
        metadata["author"] = meta_author["content"]

    # Extract robots directives
    meta_robots = soup.find("meta", attrs={"name": "robots"})
    if meta_robots and "content" in meta_robots.attrs:
        metadata["robots"] = meta_robots["content"]

    # Extract favicon
    favicon = None
    for link in soup.find_all("link", rel=re.compile(r"(icon|shortcut icon)")):
        if "href" in link.attrs:
            favicon = link["href"]
            break

    if favicon:
        metadata["favicon"] = favicon

    return metadata


def extract_opengraph(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extract Open Graph metadata.

    Args:
        soup: BeautifulSoup object

    Returns:
        Dictionary of Open Graph properties
    """
    og_data = {}

    for meta in soup.find_all("meta", property=re.compile(r"^og:")):
        if "content" in meta.attrs:
            property_name = meta["property"][3:]  # Remove 'og:' prefix
            og_data[property_name] = meta["content"]

    return og_data


def extract_microdata(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Extract microdata (schema.org) from HTML.

    Args:
        soup: BeautifulSoup object

    Returns:
        List of extracted microdata items
    """
    items = []

    # Find elements with itemscope attribute
    for element in soup.find_all(itemscope=True):
        item = {}

        # Get item type
        if element.has_attr("itemtype"):
            item["type"] = element["itemtype"]

        # Get properties
        properties = {}
        for prop in element.find_all(itemprop=True):
            prop_name = prop["itemprop"]

            # Extract value based on element type
            if prop.name == "meta":
                prop_value = prop.get("content", "")
            elif prop.name == "img":
                prop_value = prop.get("src", "")
            elif prop.name == "a":
                prop_value = prop.get("href", "")
            elif prop.name == "time":
                prop_value = prop.get("datetime", prop.get_text(strip=True))
            else:
                prop_value = prop.get_text(strip=True)

            properties[prop_name] = prop_value

        item["properties"] = properties

        # Only add if we have properties
        if properties:
            items.append(item)

    return items


def extract_structured_content(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extract structured content elements like lists, tables, and definitions.

    Args:
        soup: BeautifulSoup object

    Returns:
        Dictionary with structured content
    """
    structured_content = {
        "lists": _extract_lists(soup),
        "tables": _extract_tables(soup),
        "definitions": _extract_definitions(soup),
        "blockquotes": _extract_blockquotes(soup),
        "code_blocks": _extract_code_blocks(soup),
    }

    return structured_content


def _extract_lists(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Extract lists from the HTML document.

    Args:
        soup: BeautifulSoup object

    Returns:
        List of extracted list objects
    """
    extracted_lists = []

    # Process both ordered and unordered lists
    for list_elem in soup.find_all(["ul", "ol"]):
        # Skip empty lists
        items = list_elem.find_all("li", recursive=False)
        if not items:
            continue

        # Skip likely navigation menus
        parent = list_elem.parent
        if parent.name == "nav":
            continue

        # Skip if all items are just links (likely a menu)
        if all(
            len(li.find_all("a")) == 1 and len(li.get_text(strip=True)) < 30
            for li in items
        ):
            continue

        # Extract list items
        list_items = []
        for li in items:
            item_text = li.get_text(strip=True)

            # Skip empty items
            if not item_text:
                continue

            # Check for nested lists
            nested_lists = []
            for nested_list in li.find_all(["ul", "ol"], recursive=False):
                nested_items = [
                    nested_li.get_text(strip=True)
                    for nested_li in nested_list.find_all("li")
                    if nested_li.get_text(strip=True)
                ]
                if nested_items:
                    nested_lists.append(
                        {"type": nested_list.name, "items": nested_items}
                    )

            list_items.append(
                {
                    "text": item_text,
                    "has_link": bool(li.find("a")),
                    "nested_lists": nested_lists,
                }
            )

        # Only add if we have items
        if list_items:
            # Determine list context
            heading = None
            prev_elem = list_elem.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
            if prev_elem and len(list_elem.get_text(strip=True)) < 1000:
                # Only use heading if it's close to the list
                heading = prev_elem.get_text(strip=True)

            extracted_lists.append(
                {
                    "type": list_elem.name,  # 'ul' or 'ol'
                    "items": list_items,
                    "item_count": len(list_items),
                    "context_heading": heading,
                }
            )

    return extracted_lists


def _extract_tables(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Extract tables from the HTML document.

    Args:
        soup: BeautifulSoup object

    Returns:
        List of extracted table objects
    """
    extracted_tables = []

    for table in soup.find_all("table"):
        # Skip empty tables
        if not table.find("tr"):
            continue

        # Get caption if available
        caption = table.find("caption")
        caption_text = caption.get_text(strip=True) if caption else None

        # Extract headers
        headers = []
        thead = table.find("thead")
        if thead:
            header_cells = thead.find_all(["th", "td"])
            headers = [cell.get_text(strip=True) for cell in header_cells]

        # If no thead, try to use first row as header
        if not headers:
            first_row = table.find("tr")
            if first_row:
                header_cells = first_row.find_all(["th", "td"])
                if any(cell.name == "th" for cell in header_cells):
                    headers = [cell.get_text(strip=True) for cell in header_cells]

        # Extract data rows
        rows = []
        data_rows = table.find_all("tr")

        # Skip header row if we used it for headers
        if headers and len(data_rows) > 0 and not thead:
            data_rows = data_rows[1:]

        for row in data_rows:
            cells = row.find_all(["td", "th"])
            if cells:
                row_data = [cell.get_text(strip=True) for cell in cells]
                rows.append(row_data)

        # Only add if table has data
        if rows:
            # Determine table context
            heading = None
            prev_elem = table.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
            if prev_elem:
                heading = prev_elem.get_text(strip=True)

            extracted_tables.append(
                {
                    "caption": caption_text,
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows),
                    "column_count": (
                        len(headers) if headers else len(rows[0]) if rows else 0
                    ),
                    "context_heading": heading,
                }
            )

    return extracted_tables


def _extract_definitions(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract definition lists from the HTML document.

    Args:
        soup: BeautifulSoup object

    Returns:
        List of term-definition pairs
    """
    definitions = []

    for dl in soup.find_all("dl"):
        terms = []
        current_term = None

        for child in dl.children:
            if child.name == "dt":
                current_term = child.get_text(strip=True)
            elif child.name == "dd" and current_term:
                definition = child.get_text(strip=True)
                terms.append({"term": current_term, "definition": definition})
                current_term = None

        if terms:
            definitions.extend(terms)

    return definitions


def _extract_blockquotes(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract blockquotes from the HTML document.

    Args:
        soup: BeautifulSoup object

    Returns:
        List of quote objects
    """
    quotes = []

    for blockquote in soup.find_all("blockquote"):
        text = blockquote.get_text(strip=True)
        if not text:
            continue

        # Look for citation
        cite = blockquote.find("cite")
        citation = cite.get_text(strip=True) if cite else None

        # Check for footer citation
        if not citation:
            footer = blockquote.find("footer")
            if footer:
                citation = footer.get_text(strip=True)

        quotes.append({"text": text, "citation": citation})

    return quotes


def _extract_code_blocks(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract code blocks from the HTML document.

    Args:
        soup: BeautifulSoup object

    Returns:
        List of code block objects
    """
    code_blocks = []

    # Find <pre><code> blocks
    for pre in soup.find_all("pre"):
        code = pre.find("code")

        # Skip if it's not a code block
        if not code and not re.search(
            r"code|syntax|highlight", " ".join(pre.get("class", []))
        ):
            continue

        if code:
            content = code.get_text(strip=False)
        else:
            content = pre.get_text(strip=False)

        if not content.strip():
            continue

        # Try to determine language
        language = None

        # Check element classes for language hints
        element = code if code else pre
        classes = element.get("class", [])

        for cls in classes:
            # Common patterns: "language-python", "lang-js", "brush: php"
            if cls.startswith(("language-", "lang-")):
                language = cls.split("-", 1)[1]
                break
            elif cls.startswith("brush:"):
                language = cls.split(":", 1)[1].strip()
                break

        code_blocks.append({"content": content, "language": language})

    return code_blocks
