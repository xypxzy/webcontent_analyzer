from typing import Dict, Any, List, Optional, Set, Tuple
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse, urljoin
import re


class StructureAnalyzer:
    """
    Analyzer for page structure, identifying key elements and patterns.
    """

    def __init__(self):
        """Initialize the structure analyzer."""
        # Common form indicators
        self.form_indicators = {
            "search": ["search", "find", "query", "buscar", "поиск", "recherche"],
            "login": ["login", "signin", "log-in", "sign-in", "вход", "connexion"],
            "register": [
                "register",
                "signup",
                "sign-up",
                "create-account",
                "регистрация",
                "inscription",
            ],
            "contact": [
                "contact",
                "feedback",
                "message",
                "контакт",
                "связаться",
                "contact-us",
                "contact_us",
            ],
            "newsletter": [
                "newsletter",
                "subscribe",
                "подписаться",
                "подписка",
                "abonnement",
            ],
            "checkout": ["checkout", "payment", "оплата", "paiement", "cart"],
        }

        # Common navigation patterns
        self.nav_indicators = [
            "menu",
            "nav",
            "navigation",
            "navbar",
            "main-menu",
            "header-menu",
            "top-menu",
            "навигация",
            "меню",
        ]

    def analyze(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Analyze page structure and return detailed information.

        Args:
            soup: BeautifulSoup object
            url: URL of the page

        Returns:
            Dictionary containing structure information
        """
        structure = {
            # Document structure
            "headings": self._analyze_headings(soup),
            "sections": self._identify_sections(soup),
            "content_blocks": self._identify_content_blocks(soup),
            # Navigation elements
            "navigation": self._analyze_navigation(soup),
            "breadcrumbs": self._extract_breadcrumbs(soup),
            # Links and references
            "links": self._analyze_links(soup, url),
            "internal_links": self._analyze_internal_links(soup, url),
            "external_links": self._analyze_external_links(soup, url),
            # Interactions
            "forms": self._analyze_forms(soup),
            "cta_elements": self._identify_cta_elements(soup),
            "search_functionality": self._identify_search(soup),
            # Media
            "images": self._analyze_images(soup),
            "videos": self._identify_videos(soup),
            # Layout elements
            "layout": self._analyze_layout(soup),
            "sidebars": self._identify_sidebars(soup),
            "footers": self._identify_footer(soup),
            "headers": self._identify_header(soup),
            # Content elements
            "lists": self._analyze_lists(soup),
            "tables": self._analyze_tables(soup),
            # Technical structure
            "iframe_usage": self._analyze_iframes(soup),
            "javascript_dependencies": self._analyze_scripts(soup),
            "css_dependencies": self._analyze_stylesheets(soup),
            "page_complexity": self._calculate_complexity(soup),
        }

        return structure

    def _analyze_headings(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze headings structure and hierarchy.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with heading analysis
        """
        heading_counts = {}
        headings_content = {}
        heading_structure = []
        has_proper_hierarchy = True
        last_level = 0

        # Analyze all headings
        for heading_tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            headings = soup.find_all(heading_tag)
            count = len(headings)
            heading_counts[heading_tag] = count

            # Store heading content
            if count > 0:
                headings_content[heading_tag] = [h.get_text().strip() for h in headings]

                # Check for proper hierarchy
                level = int(heading_tag[1])
                if level > last_level + 1 and last_level > 0:
                    has_proper_hierarchy = False
                if count > 0:
                    last_level = level

                # Build structure
                for h in headings:
                    heading_structure.append(
                        {
                            "level": level,
                            "text": h.get_text().strip(),
                            "id": h.get("id", ""),
                            "classes": h.get("class", []),
                        }
                    )

        # Calculate uniqueness of headings
        unique_h1 = len(set(headings_content.get("h1", [])))
        unique_headings = len(
            set([h for level in headings_content.values() for h in level])
        )
        total_headings = sum(heading_counts.values())

        return {
            "counts": heading_counts,
            "total": total_headings,
            "has_h1": heading_counts.get("h1", 0) > 0,
            "multiple_h1": heading_counts.get("h1", 0) > 1,
            "has_proper_hierarchy": has_proper_hierarchy,
            "structure": heading_structure,
            "uniqueness_ratio": unique_headings / max(total_headings, 1),
            "h1_uniqueness": unique_h1 == heading_counts.get("h1", 0),
        }

    def _identify_sections(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Identify major sections of the document.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of section information
        """
        sections = []

        # Look for semantic sectioning elements
        for section_tag in ["section", "article", "main", "aside"]:
            elements = soup.find_all(section_tag)

            for elem in elements:
                # Get heading if present
                heading = elem.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                heading_text = heading.get_text().strip() if heading else ""

                # Get section information
                section_info = {
                    "type": section_tag,
                    "id": elem.get("id", ""),
                    "classes": elem.get("class", []),
                    "heading": heading_text,
                    "word_count": len(elem.get_text().split()),
                    "has_images": len(elem.find_all("img")) > 0,
                    "has_links": len(elem.find_all("a")) > 0,
                    "content_snippet": (
                        elem.get_text()[:100].strip() + "..."
                        if len(elem.get_text()) > 100
                        else elem.get_text().strip()
                    ),
                }

                sections.append(section_info)

        # Look for div-based sections with section-like classes or IDs
        section_patterns = [
            "section",
            "content",
            "block",
            "container",
            "wrapper",
            "module",
        ]
        for div in soup.find_all("div", id=True):
            if any(
                pattern in div.get("id", "").lower() for pattern in section_patterns
            ):
                heading = div.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                heading_text = heading.get_text().strip() if heading else ""

                section_info = {
                    "type": "div",
                    "id": div.get("id", ""),
                    "classes": div.get("class", []),
                    "heading": heading_text,
                    "word_count": len(div.get_text().split()),
                    "has_images": len(div.find_all("img")) > 0,
                    "has_links": len(div.find_all("a")) > 0,
                    "content_snippet": (
                        div.get_text()[:100].strip() + "..."
                        if len(div.get_text()) > 100
                        else div.get_text().strip()
                    ),
                }

                sections.append(section_info)

        return sections

    def _identify_content_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Identify content blocks within the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of content block information
        """
        blocks = []

        # Functions to calculate key metrics
        def calculate_text_density(element):
            html_length = len(str(element))
            text_length = len(element.get_text())
            if html_length == 0:
                return 0
            return text_length / html_length

        def calculate_link_density(element):
            text_length = len(element.get_text())
            link_text_length = sum(len(a.get_text()) for a in element.find_all("a"))
            if text_length == 0:
                return 0
            return link_text_length / text_length

        # Find potential content blocks
        block_candidates = []

        # Look for elements with significant text
        for elem in soup.find_all(["div", "section", "article"]):
            text = elem.get_text().strip()
            if len(text) < 100:  # Skip small blocks
                continue

            # Calculate metrics
            word_count = len(text.split())
            text_density = calculate_text_density(elem)
            link_density = calculate_link_density(elem)

            # Skip likely navigation elements
            if link_density > 0.5:
                continue

            block_candidates.append(
                {
                    "element": elem,
                    "word_count": word_count,
                    "text_density": text_density,
                    "link_density": link_density,
                }
            )

        # Score and filter candidates
        for candidate in block_candidates:
            elem = candidate["element"]

            # Skip nested blocks (keep only the outermost)
            is_nested = False
            for other in block_candidates:
                if other["element"] is not elem and elem in other["element"].find_all():
                    is_nested = True
                    break

            if is_nested:
                continue

            # Create block information
            block_info = {
                "tag": elem.name,
                "id": elem.get("id", ""),
                "classes": elem.get("class", []),
                "word_count": candidate["word_count"],
                "text_density": candidate["text_density"],
                "link_density": candidate["link_density"],
                "has_heading": bool(elem.find(["h1", "h2", "h3", "h4", "h5", "h6"])),
                "has_images": bool(elem.find_all("img")),
                "has_lists": bool(elem.find_all(["ul", "ol"])),
                "content_snippet": (
                    elem.get_text()[:100].strip() + "..."
                    if len(elem.get_text()) > 100
                    else elem.get_text().strip()
                ),
            }

            blocks.append(block_info)

        return blocks

    def _analyze_navigation(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze navigation elements.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with navigation analysis
        """
        # Find navigation elements
        nav_elements = []

        # Look for semantic nav elements
        for nav in soup.find_all("nav"):
            nav_elements.append(
                {
                    "type": "semantic",
                    "id": nav.get("id", ""),
                    "classes": nav.get("class", []),
                    "location": self._determine_element_location(nav),
                    "link_count": len(nav.find_all("a")),
                    "text": [
                        a.get_text().strip()
                        for a in nav.find_all("a")
                        if a.get_text().strip()
                    ],
                    "has_dropdown": bool(nav.select("ul ul"))
                    or bool(nav.select(".dropdown, .submenu")),
                }
            )

        # Look for non-semantic navigation elements
        for elem in soup.find_all(["div", "ul"]):
            # Check for navigation classes or IDs
            elem_id = elem.get("id", "").lower()
            elem_classes = " ".join(elem.get("class", [])).lower()

            if any(nav_id in elem_id for nav_id in self.nav_indicators) or any(
                nav_class in elem_classes for nav_class in self.nav_indicators
            ):
                # Found potential navigation
                links = elem.find_all("a")
                if len(links) < 2:  # Skip elements with very few links
                    continue

                nav_elements.append(
                    {
                        "type": "non-semantic",
                        "id": elem.get("id", ""),
                        "classes": elem.get("class", []),
                        "location": self._determine_element_location(elem),
                        "link_count": len(links),
                        "text": [
                            a.get_text().strip() for a in links if a.get_text().strip()
                        ],
                        "has_dropdown": bool(elem.select("ul ul"))
                        or bool(elem.select(".dropdown, .submenu")),
                    }
                )

        # Identify main navigation
        main_nav = None
        if nav_elements:
            # Sort by link count and position
            main_candidates = sorted(
                nav_elements,
                key=lambda x: (
                    x["location"] == "header",
                    x["has_dropdown"],
                    x["link_count"],
                ),
                reverse=True,
            )

            if main_candidates:
                main_nav = main_candidates[0]

        return {
            "nav_elements": nav_elements,
            "nav_count": len(nav_elements),
            "has_semantic_nav": any(n["type"] == "semantic" for n in nav_elements),
            "main_navigation": main_nav,
            "has_dropdown_menus": any(n["has_dropdown"] for n in nav_elements),
            "link_distribution": {
                "header": sum(1 for n in nav_elements if n["location"] == "header"),
                "footer": sum(1 for n in nav_elements if n["location"] == "footer"),
                "sidebar": sum(1 for n in nav_elements if n["location"] == "sidebar"),
                "other": sum(
                    1
                    for n in nav_elements
                    if n["location"] not in ["header", "footer", "sidebar"]
                ),
            },
        }

    def _extract_breadcrumbs(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract breadcrumb navigation if present.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with breadcrumb information
        """
        breadcrumb_info = {"present": False, "items": [], "schema_markup": False}

        # Check for schema.org breadcrumb markup
        schema_breadcrumb = soup.find(
            ["ol", "ul", "nav"], itemtype=lambda x: x and "BreadcrumbList" in x
        )

        if schema_breadcrumb:
            breadcrumb_info["present"] = True
            breadcrumb_info["schema_markup"] = True

            items = []
            for item in schema_breadcrumb.find_all(itemprop="itemListElement"):
                name_elem = item.find(itemprop="name")
                url_elem = item.find(itemprop="item")

                if name_elem:
                    item_info = {
                        "text": name_elem.get_text().strip(),
                        "url": url_elem.get("href", "") if url_elem else "",
                    }
                    items.append(item_info)

            breadcrumb_info["items"] = items
            return breadcrumb_info

        # Check for common breadcrumb patterns
        breadcrumb_candidates = []

        # Look for elements with breadcrumb-related classes or IDs
        for elem in soup.find_all(["ol", "ul", "nav", "div"]):
            elem_id = elem.get("id", "").lower()
            elem_classes = " ".join(elem.get("class", [])).lower()

            if "breadcrumb" in elem_id or "breadcrumb" in elem_classes:
                breadcrumb_candidates.append(elem)

        # If no explicit breadcrumb element found, try common patterns
        if not breadcrumb_candidates:
            # Look for small lists of links (3-5) that might be breadcrumbs
            for elem in soup.find_all(["ol", "ul"]):
                links = elem.find_all("a")

                # Breadcrumbs typically have a small number of links
                if 2 <= len(links) <= 5:
                    # Check for separator characters between links
                    text = elem.get_text()
                    if ">" in text or "/" in text or "»" in text:
                        breadcrumb_candidates.append(elem)

        # Process candidates
        if breadcrumb_candidates:
            # Use the first candidate (most likely breadcrumb)
            breadcrumb = breadcrumb_candidates[0]

            # Extract links
            links = breadcrumb.find_all("a")
            items = []

            for link in links:
                item_info = {
                    "text": link.get_text().strip(),
                    "url": link.get("href", ""),
                }
                items.append(item_info)

            # Check for current item (often not a link)
            current_item = None

            # Check for special classes that indicate current item
            current_candidates = breadcrumb.select(".active, .current")
            if current_candidates:
                current_item = current_candidates[0].get_text().strip()
            else:
                # Last item might be plain text (not a link)
                for child in list(breadcrumb.children):
                    if (
                        child.name is None
                        and child.strip()
                        and child.strip() not in [",", "/", ">", "»", "|"]
                    ):
                        current_item = child.strip()
                        break

            # If found a current item, add it
            if current_item and current_item not in [item["text"] for item in items]:
                items.append({"text": current_item, "url": "", "current": True})

            breadcrumb_info["present"] = True
            breadcrumb_info["items"] = items

        return breadcrumb_info

    def _analyze_links(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Analyze links on the page.

        Args:
            soup: BeautifulSoup object
            url: URL of the page

        Returns:
            Dictionary with link analysis
        """
        links = soup.find_all("a")

        # Count links
        total_links = len(links)
        links_with_text = sum(1 for link in links if link.get_text().strip())

        # Analyze link text
        link_texts = [
            link.get_text().strip() for link in links if link.get_text().strip()
        ]
        unique_link_texts = len(set(link_texts))
        average_link_length = sum(len(text) for text in link_texts) / max(
            len(link_texts), 1
        )

        # Analyze hrefs
        valid_hrefs = [link.get("href", "") for link in links if link.get("href")]
        empty_hrefs = sum(1 for link in links if not link.get("href"))

        # Count different link types
        link_types = {
            "anchor": 0,
            "internal": 0,
            "external": 0,
            "mailto": 0,
            "tel": 0,
            "javascript": 0,
            "file": 0,
        }

        base_domain = urlparse(url).netloc

        for link in links:
            href = link.get("href", "")
            if not href:
                continue

            if href.startswith("#"):
                link_types["anchor"] += 1
            elif href.startswith("mailto:"):
                link_types["mailto"] += 1
            elif href.startswith("tel:"):
                link_types["tel"] += 1
            elif href.startswith("javascript:"):
                link_types["javascript"] += 1
            else:
                full_url = urljoin(url, href)
                parsed_url = urlparse(full_url)

                if parsed_url.netloc == base_domain or not parsed_url.netloc:
                    link_types["internal"] += 1
                else:
                    link_types["external"] += 1

                # Check if it's a file download
                if re.search(
                    r"\.(pdf|doc|docx|xls|xlsx|zip|rar|tar|gz|mp3|mp4)$",
                    parsed_url.path,
                    re.I,
                ):
                    link_types["file"] += 1

        # Analyze link distribution
        link_distribution = {}

        for element_type in ["header", "footer", "sidebar", "nav", "main", "article"]:
            elements = soup.find_all(element_type)
            if elements:
                link_count = sum(len(elem.find_all("a")) for elem in elements)
                link_distribution[element_type] = link_count

        return {
            "total_links": total_links,
            "links_with_text": links_with_text,
            "empty_links": empty_hrefs,
            "unique_link_texts": unique_link_texts,
            "link_text_diversity": unique_link_texts / max(links_with_text, 1),
            "average_link_text_length": average_link_length,
            "link_types": link_types,
            "link_distribution": link_distribution,
        }

    def _analyze_internal_links(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Analyze internal links on the page.

        Args:
            soup: BeautifulSoup object
            url: URL of the page

        Returns:
            Dictionary with internal link analysis
        """
        base_domain = urlparse(url).netloc
        base_path = urlparse(url).path

        # Collect internal links
        internal_links = []

        for link in soup.find_all("a", href=True):
            href = link.get("href", "")

            # Skip non-internal links
            if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue

            # Resolve URL
            full_url = urljoin(url, href)
            parsed_url = urlparse(full_url)

            # Check if internal
            if parsed_url.netloc == base_domain or not parsed_url.netloc:
                link_text = link.get_text().strip()

                # Extract additional attributes
                rel = link.get("rel", [])
                title = link.get("title", "")

                internal_links.append(
                    {
                        "url": full_url,
                        "path": parsed_url.path,
                        "text": link_text,
                        "same_page": base_path == parsed_url.path,
                        "nofollow": "nofollow" in rel,
                        "title": title,
                        "has_image": bool(link.find("img")),
                    }
                )

        # Analyze section linking
        section_links = sum(
            1 for link in internal_links if link["url"].startswith(url + "#")
        )

        # Find most linked sections
        path_frequency = {}
        for link in internal_links:
            if link["path"] in path_frequency:
                path_frequency[link["path"]] += 1
            else:
                path_frequency[link["path"]] = 1

        # Sort paths by frequency
        popular_paths = sorted(
            path_frequency.items(), key=lambda x: x[1], reverse=True
        )[
            :5
        ]  # Top 5 most linked paths

        return {
            "count": len(internal_links),
            "unique_urls": len(set(link["url"] for link in internal_links)),
            "section_links": section_links,
            "popular_paths": popular_paths,
            "nofollow_count": sum(1 for link in internal_links if link["nofollow"]),
            "links_with_text": sum(1 for link in internal_links if link["text"]),
            "links_with_images": sum(1 for link in internal_links if link["has_image"]),
            "linked_sections": [
                link
                for link in internal_links
                if link["same_page"] and "#" in link["url"]
            ],
        }

    def _analyze_external_links(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Analyze external links on the page.

        Args:
            soup: BeautifulSoup object
            url: URL of the page

        Returns:
            Dictionary with external link analysis
        """
        base_domain = urlparse(url).netloc

        # Collect external links
        external_links = []

        for link in soup.find_all("a", href=True):
            href = link.get("href", "")

            # Skip non-URL links
            if href.startswith(("#", "javascript:")):
                continue

            # Resolve URL
            full_url = urljoin(url, href)
            parsed_url = urlparse(full_url)

            # Check if external
            if parsed_url.netloc and parsed_url.netloc != base_domain:
                link_text = link.get_text().strip()

                # Extract additional attributes
                rel = link.get("rel", [])
                title = link.get("title", "")

                external_links.append(
                    {
                        "url": full_url,
                        "domain": parsed_url.netloc,
                        "text": link_text,
                        "nofollow": "nofollow" in rel,
                        "title": title,
                        "has_image": bool(link.find("img")),
                        "is_social": self._is_social_media_link(full_url),
                    }
                )

        # Analyze domains
        domains = {}
        for link in external_links:
            domain = link["domain"]
            if domain in domains:
                domains[domain] += 1
            else:
                domains[domain] = 1

        # Sort domains by frequency
        popular_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]  # Top 5 most linked domains

        # Count social media links
        social_links = sum(1 for link in external_links if link["is_social"])

        return {
            "count": len(external_links),
            "unique_domains": len(domains),
            "popular_domains": popular_domains,
            "nofollow_count": sum(1 for link in external_links if link["nofollow"]),
            "social_links": social_links,
            "links_with_text": sum(1 for link in external_links if link["text"]),
            "links_with_images": sum(1 for link in external_links if link["has_image"]),
        }

    def _analyze_forms(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze forms on the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with form analysis
        """
        forms = soup.find_all("form")
        form_analysis = []

        form_types = {
            "search": 0,
            "login": 0,
            "register": 0,
            "contact": 0,
            "newsletter": 0,
            "checkout": 0,
            "other": 0,
        }

        for form in forms:
            # Get form attributes
            form_id = form.get("id", "")
            form_classes = " ".join(form.get("class", []))
            form_action = form.get("action", "")
            form_method = form.get("method", "get").lower()

            # Analyze form inputs
            inputs = form.find_all(["input", "textarea", "select"])
            input_types = {}

            for inp in inputs:
                if inp.name == "input":
                    input_type = inp.get("type", "text").lower()
                    if input_type in input_types:
                        input_types[input_type] += 1
                    else:
                        input_types[input_type] = 1
                else:
                    if inp.name in input_types:
                        input_types[inp.name] += 1
                    else:
                        input_types[inp.name] = 1

            # Determine form type
            form_type = "other"
            id_and_classes = (form_id + " " + form_classes).lower()

            for type_name, indicators in self.form_indicators.items():
                if any(indicator in id_and_classes for indicator in indicators):
                    form_type = type_name
                    break

            # If not determined by class/id, try to determine by content
            if form_type == "other":
                # Check for search form
                if input_types.get("search", 0) > 0 or "search" in form_action.lower():
                    form_type = "search"
                # Check for login form
                elif (
                    input_types.get("password", 0) > 0
                    and input_types.get("email", 0) + input_types.get("text", 0) > 0
                ):
                    form_type = "login"
                # Check for registration form
                elif input_types.get("password", 0) > 1 or (
                    input_types.get("password", 0) > 0 and len(inputs) > 3
                ):
                    form_type = "register"
                # Check for contact form
                elif (
                    input_types.get("email", 0) > 0
                    and input_types.get("textarea", 0) > 0
                ):
                    form_type = "contact"
                # Check for newsletter form
                elif input_types.get("email", 0) > 0 and len(inputs) < 3:
                    form_type = "newsletter"

            # Increment form type counter
            form_types[form_type] += 1

            # Check if form has submit button
            submit_buttons = form.find_all(["button", "input"], type="submit")
            submit_text = ""
            if submit_buttons:
                submit_text = submit_buttons[0].get_text().strip() or submit_buttons[
                    0
                ].get("value", "")

            # Check for required fields
            required_fields = form.find_all(required=True)

            form_analysis.append(
                {
                    "type": form_type,
                    "method": form_method,
                    "action": form_action,
                    "id": form_id,
                    "classes": form.get("class", []),
                    "field_count": len(inputs),
                    "input_types": input_types,
                    "has_submit": bool(submit_buttons),
                    "submit_text": submit_text,
                    "required_fields": len(required_fields),
                }
            )

        return {
            "count": len(forms),
            "forms": form_analysis,
            "form_types": form_types,
            "post_forms": sum(1 for f in form_analysis if f["method"] == "post"),
            "get_forms": sum(1 for f in form_analysis if f["method"] == "get"),
            "avg_fields_per_form": sum(f["field_count"] for f in form_analysis)
            / max(len(forms), 1),
        }

    def _identify_cta_elements(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Identify call-to-action elements on the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of CTA elements
        """
        cta_elements = []

        # Find buttons with CTA-like text
        cta_text_patterns = [
            "sign up",
            "register",
            "subscribe",
            "join",
            "get started",
            "try",
            "buy",
            "purchase",
            "order",
            "add to cart",
            "checkout",
            "download",
            "request",
            "contact",
            "learn more",
            "read more",
            "more info",
            "continue",
        ]

        # Look for buttons
        for button in soup.find_all(
            ["button", "a"], class_=re.compile(r"btn|button", re.I)
        ):
            text = button.get_text().strip().lower()

            if text and (
                any(pattern in text for pattern in cta_text_patterns) or len(text) < 20
            ):
                cta_elements.append(
                    {
                        "type": "button",
                        "element": button.name,
                        "text": button.get_text().strip(),
                        "url": button.get("href", "") if button.name == "a" else "",
                        "classes": button.get("class", []),
                        "location": self._determine_element_location(button),
                        "is_primary": any(
                            cls in " ".join(button.get("class", [])).lower()
                            for cls in ["primary", "main", "cta"]
                        ),
                        "has_icon": bool(
                            button.find("i") or button.find("svg") or button.find("img")
                        ),
                    }
                )

        # Look for elements with CTA-like classes
        cta_classes = [
            "cta",
            "call-to-action",
            "action",
            "primary-button",
            "main-button",
        ]
        for elem in soup.find_all(
            class_=lambda c: c and any(cta_cls in c.lower() for cta_cls in cta_classes)
        ):
            # Skip if already included as a button
            if elem.name in ["button", "a"] and any(
                cta["element"] == elem.name and cta["text"] == elem.get_text().strip()
                for cta in cta_elements
            ):
                continue

            text = elem.get_text().strip()
            if text:
                cta_elements.append(
                    {
                        "type": "element_with_cta_class",
                        "element": elem.name,
                        "text": text,
                        "url": elem.get("href", "") if elem.name == "a" else "",
                        "classes": elem.get("class", []),
                        "location": self._determine_element_location(elem),
                        "is_primary": True,  # If it has a CTA class, consider it primary
                        "has_icon": bool(
                            elem.find("i") or elem.find("svg") or elem.find("img")
                        ),
                    }
                )

        # Look for forms with CTA-like submit buttons
        for form in soup.find_all("form"):
            submit_buttons = form.find_all(["button", "input"], type="submit")
            if not submit_buttons:
                continue

            submit_text = submit_buttons[0].get_text().strip() or submit_buttons[0].get(
                "value", ""
            )

            if submit_text and any(
                pattern in submit_text.lower() for pattern in cta_text_patterns
            ):
                # Determine form type
                form_id = form.get("id", "")
                form_classes = " ".join(form.get("class", []))
                id_and_classes = (form_id + " " + form_classes).lower()

                form_type = "other"
                for type_name, indicators in self.form_indicators.items():
                    if any(indicator in id_and_classes for indicator in indicators):
                        form_type = type_name
                        break

                cta_elements.append(
                    {
                        "type": "form_submit",
                        "element": "form",
                        "text": submit_text,
                        "form_type": form_type,
                        "classes": form.get("class", []),
                        "location": self._determine_element_location(form),
                        "is_primary": any(
                            cls in " ".join(submit_buttons[0].get("class", [])).lower()
                            for cls in ["primary", "main", "cta"]
                        ),
                        "has_icon": bool(
                            submit_buttons[0].find("i")
                            or submit_buttons[0].find("svg")
                            or submit_buttons[0].find("img")
                        ),
                    }
                )

        return cta_elements

    def _identify_search(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Identify search functionality on the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with search functionality information
        """
        search_info = {
            "present": False,
            "type": None,
            "location": None,
            "placeholder": None,
            "button_text": None,
        }

        # Look for search forms
        search_forms = []

        # Check form with search-related attributes
        for form in soup.find_all("form"):
            form_id = form.get("id", "").lower()
            form_classes = " ".join(form.get("class", [])).lower()
            form_action = form.get("action", "").lower()

            # Check form attributes for search indicators
            if (
                "search" in form_id
                or "search" in form_classes
                or "search" in form_action
                or "find" in form_id
                or "find" in form_classes
            ):
                search_forms.append(form)

        # Check for search input fields
        search_inputs = soup.find_all("input", attrs={"type": "search"})

        search_inputs.extend(
            soup.find_all(
                "input", attrs={"name": re.compile(r"search|query|q\b", re.I)}
            )
        )

        search_inputs.extend(
            soup.find_all(
                "input",
                attrs={"placeholder": re.compile(r"search|find|look for", re.I)},
            )
        )

        # Process found search elements
        if search_forms or search_inputs:
            search_info["present"] = True

            # Determine search type
            if search_forms:
                search_form = search_forms[0]
                search_info["type"] = "form"
                search_info["location"] = self._determine_element_location(search_form)

                # Find search input within form
                search_input = search_form.find(
                    "input",
                    attrs={
                        "type": ["search", "text"],
                        "name": re.compile(r"search|query|q\b", re.I),
                    },
                )

                if search_input:
                    search_info["placeholder"] = search_input.get("placeholder", "")

                # Find search button
                search_button = search_form.find(
                    ["button", "input"], attrs={"type": ["submit", "button"]}
                )

                if search_button:
                    if search_button.name == "button":
                        search_info["button_text"] = search_button.get_text().strip()
                    else:
                        search_info["button_text"] = search_button.get("value", "")

            elif search_inputs:
                search_input = search_inputs[0]
                search_info["type"] = "input"
                search_info["location"] = self._determine_element_location(search_input)
                search_info["placeholder"] = search_input.get("placeholder", "")

                # Look for adjacent button
                parent = search_input.parent
                search_button = parent.find(
                    ["button", "input"], attrs={"type": ["submit", "button"]}
                )

                if search_button:
                    if search_button.name == "button":
                        search_info["button_text"] = search_button.get_text().strip()
                    else:
                        search_info["button_text"] = search_button.get("value", "")

        return search_info

    def _analyze_images(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze images on the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with image analysis
        """
        images = soup.find_all("img")

        # Initialize counters
        image_types = {
            "with_alt": 0,
            "empty_alt": 0,
            "missing_alt": 0,
            "linked": 0,
            "responsive": 0,
            "lazy_loaded": 0,
        }

        image_analysis = []

        for img in images:
            # Get attributes
            src = img.get("src", "")
            alt = img.get("alt", None)

            # Check if image is linked
            is_linked = img.parent.name == "a"

            # Check for responsive images
            is_responsive = (
                "class" in img.attrs
                and any("responsive" in cls.lower() for cls in img.get("class", []))
                or img.has_attr("srcset")
                or "sizes" in img.attrs
                or "width" in img.attrs
                and "height" in img.attrs
            )

            # Check for lazy loading
            is_lazy_loaded = (
                img.has_attr("loading")
                and img["loading"] == "lazy"
                or img.has_attr("data-src")
                or "class" in img.attrs
                and any("lazy" in cls.lower() for cls in img.get("class", []))
            )

            # Update counters
            if alt is not None:
                if alt.strip():
                    image_types["with_alt"] += 1
                else:
                    image_types["empty_alt"] += 1
            else:
                image_types["missing_alt"] += 1

            if is_linked:
                image_types["linked"] += 1

            if is_responsive:
                image_types["responsive"] += 1

            if is_lazy_loaded:
                image_types["lazy_loaded"] += 1

            # Add to detailed analysis
            if src:  # Skip images without src
                image_analysis.append(
                    {
                        "src": src,
                        "alt": alt or "",
                        "width": img.get("width", ""),
                        "height": img.get("height", ""),
                        "is_linked": is_linked,
                        "link_url": img.parent.get("href", "") if is_linked else "",
                        "is_responsive": is_responsive,
                        "is_lazy_loaded": is_lazy_loaded,
                        "location": self._determine_element_location(img),
                    }
                )

        return {
            "count": len(images),
            "image_types": image_types,
            "alt_text_ratio": image_types["with_alt"] / max(len(images), 1),
            "linked_ratio": image_types["linked"] / max(len(images), 1),
            "responsive_ratio": image_types["responsive"] / max(len(images), 1),
            "lazy_loaded_ratio": image_types["lazy_loaded"] / max(len(images), 1),
            "images": image_analysis[:20],  # Limit to 20 images for brevity
        }

    def _identify_videos(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Identify videos on the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of video elements
        """
        videos = []

        # Check for HTML5 video tags
        for video in soup.find_all("video"):
            video_info = {
                "type": "html5",
                "src": video.get("src", ""),
                "sources": [
                    source.get("src", "") for source in video.find_all("source")
                ],
                "controls": video.has_attr("controls"),
                "autoplay": video.has_attr("autoplay"),
                "loop": video.has_attr("loop"),
                "muted": video.has_attr("muted"),
                "poster": video.get("poster", ""),
                "location": self._determine_element_location(video),
            }
            videos.append(video_info)

        # Check for YouTube embeds
        youtube_iframes = soup.find_all(
            "iframe", src=re.compile(r"youtube\.com|youtu\.be", re.I)
        )
        for iframe in youtube_iframes:
            src = iframe.get("src", "")
            match = re.search(
                r"(?:youtu\.be/|youtube\.com/embed/|youtube\.com/watch\?v=)([^&?/]+)",
                src,
            )
            video_id = match.group(1) if match else ""

            video_info = {
                "type": "youtube",
                "src": src,
                "video_id": video_id,
                "width": iframe.get("width", ""),
                "height": iframe.get("height", ""),
                "location": self._determine_element_location(iframe),
            }
            videos.append(video_info)

        # Check for Vimeo embeds
        vimeo_iframes = soup.find_all("iframe", src=re.compile(r"vimeo\.com", re.I))
        for iframe in vimeo_iframes:
            src = iframe.get("src", "")
            match = re.search(r"vimeo\.com/(?:video/)?([0-9]+)", src)
            video_id = match.group(1) if match else ""

            video_info = {
                "type": "vimeo",
                "src": src,
                "video_id": video_id,
                "width": iframe.get("width", ""),
                "height": iframe.get("height", ""),
                "location": self._determine_element_location(iframe),
            }
            videos.append(video_info)

        # Check for video.js implementations
        videojs_elements = soup.find_all(class_="video-js")
        for elem in videojs_elements:
            sources = elem.find_all("source")

            video_info = {
                "type": "video.js",
                "sources": [source.get("src", "") for source in sources],
                "data_setup": elem.get("data-setup", ""),
                "poster": elem.get("poster", ""),
                "location": self._determine_element_location(elem),
            }
            videos.append(video_info)

        return videos

    def _analyze_layout(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze page layout structure.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with layout analysis
        """
        # Check for common layout indicators
        layout_info = {
            "has_header": bool(
                soup.find("header")
                or soup.find(id="header")
                or soup.find(class_=re.compile(r"header|main-header", re.I))
            ),
            "has_footer": bool(
                soup.find("footer")
                or soup.find(id="footer")
                or soup.find(class_=re.compile(r"footer|main-footer", re.I))
            ),
            "has_main": bool(
                soup.find("main")
                or soup.find(id="main")
                or soup.find(class_=re.compile(r"main-content|main-container", re.I))
            ),
            "has_sidebar": bool(
                soup.find("aside")
                or soup.find(id=re.compile(r"sidebar", re.I))
                or soup.find(class_=re.compile(r"sidebar|side-column", re.I))
            ),
            "has_navigation": bool(
                soup.find("nav")
                or soup.find(id=re.compile(r"nav|menu", re.I))
                or soup.find(class_=re.compile(r"nav|menu|navigation", re.I))
            ),
        }

        # Detect grid-based layouts
        layout_info["has_grid"] = bool(
            soup.select(".row, .grid, .container .row, .container-fluid .row")
            or soup.find_all(class_=re.compile(r"col-|grid-|span\d+", re.I))
        )

        # Detect flexbox layouts
        layout_info["has_flexbox"] = bool(
            soup.find_all(class_=re.compile(r"flex|d-flex", re.I))
        )

        # Detect common CSS frameworks
        framework_indicators = {
            "bootstrap": [
                ".container",
                ".row",
                ".col-",
                ".navbar",
                ".card",
                ".btn-primary",
            ],
            "foundation": [".grid-x", ".cell", ".button", ".top-bar", ".callout"],
            "bulma": [".columns", ".column", ".navbar", ".button", ".box", ".hero"],
            "materialize": [
                ".container",
                ".row",
                ".col .s",
                ".card",
                ".btn",
                ".nav-wrapper",
            ],
            "tailwind": [
                'class="[^"]*flex[^"]*"',
                'class="[^"]*grid[^"]*"',
                'class="[^"]*text-[a-z]+[0-9]+[^"]*"',
                'class="[^"]*bg-[a-z]+[0-9]+[^"]*"',
            ],
        }

        detected_frameworks = []
        html_str = str(soup)

        for framework, indicators in framework_indicators.items():
            matches = sum(
                1 for indicator in indicators if re.search(indicator, html_str)
            )
            if matches >= len(indicators) // 2:
                detected_frameworks.append(framework)

        layout_info["css_frameworks"] = detected_frameworks

        # Analyze sectioning
        sections_count = len(soup.find_all("section"))
        articles_count = len(soup.find_all("article"))

        layout_info["uses_semantic_sectioning"] = (
            sections_count > 0 or articles_count > 0
        )
        layout_info["sections_count"] = sections_count
        layout_info["articles_count"] = articles_count

        # Check for layout-affecting meta tags
        viewport_meta = soup.find("meta", attrs={"name": "viewport"})
        layout_info["has_viewport_meta"] = bool(viewport_meta)

        if viewport_meta and "content" in viewport_meta.attrs:
            content = viewport_meta["content"].lower()
            layout_info["is_mobile_optimized"] = (
                "width=device-width" in content and "initial-scale=1" in content
            )
        else:
            layout_info["is_mobile_optimized"] = False

        return layout_info

    def _identify_sidebars(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Identify sidebar elements.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of sidebar information
        """
        sidebars = []

        # Look for semantic aside elements
        for aside in soup.find_all("aside"):
            sidebar_info = {
                "type": "semantic",
                "id": aside.get("id", ""),
                "classes": aside.get("class", []),
                "position": self._determine_sidebar_position(aside),
                "content_types": self._analyze_sidebar_content(aside),
            }
            sidebars.append(sidebar_info)

        # Look for non-semantic sidebar elements
        sidebar_indicators = ["sidebar", "side-col", "side-bar", "rightbar", "leftbar"]

        for elem in soup.find_all(["div", "section"]):
            elem_id = elem.get("id", "").lower()
            elem_classes = " ".join(elem.get("class", [])).lower()

            if any(indicator in elem_id for indicator in sidebar_indicators) or any(
                indicator in elem_classes for indicator in sidebar_indicators
            ):
                sidebar_info = {
                    "type": "non-semantic",
                    "id": elem.get("id", ""),
                    "classes": elem.get("class", []),
                    "position": self._determine_sidebar_position(elem),
                    "content_types": self._analyze_sidebar_content(elem),
                }
                sidebars.append(sidebar_info)

        return sidebars

    def _analyze_sidebar_content(self, sidebar: Tag) -> Dict[str, int]:
        """
        Analyze the content types in a sidebar.

        Args:
            sidebar: BeautifulSoup Tag of the sidebar

        Returns:
            Dictionary with content type counts
        """
        content_types = {
            "links": len(sidebar.find_all("a")),
            "widgets": len(sidebar.find_all(class_=re.compile(r"widget", re.I))),
            "navigation": len(sidebar.find_all(["nav", "ul", "ol"])),
            "images": len(sidebar.find_all("img")),
            "forms": len(sidebar.find_all("form")),
            "headings": len(sidebar.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])),
            "paragraphs": len(sidebar.find_all("p")),
        }

        return content_types

    def _determine_sidebar_position(self, sidebar: Tag) -> str:
        """
        Determine the likely position of a sidebar.

        Args:
            sidebar: BeautifulSoup Tag of the sidebar

        Returns:
            Position string ('left', 'right', 'unknown')
        """
        elem_id = sidebar.get("id", "").lower()
        elem_classes = " ".join(sidebar.get("class", [])).lower()

        # Check for position indicators in ID and classes
        if "left" in elem_id or "left" in elem_classes:
            return "left"
        elif "right" in elem_id or "right" in elem_classes:
            return "right"

        # Check common naming patterns
        if sidebar.has_attr("class"):
            for cls in sidebar["class"]:
                if cls.endswith("-l") or cls.startswith("l-"):
                    return "left"
                elif cls.endswith("-r") or cls.startswith("r-"):
                    return "right"

        return "unknown"

    def _identify_footer(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Identify and analyze footer section.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with footer information
        """
        footer_info = {
            "present": False,
            "semantic": False,
            "id": "",
            "classes": [],
            "content": {},
        }

        # Look for semantic footer
        footer = soup.find("footer")

        # If not found, look for common footer patterns
        if not footer:
            footer_candidates = []

            # Check by ID
            footer_by_id = soup.find(id=re.compile(r"footer", re.I))
            if footer_by_id:
                footer_candidates.append(footer_by_id)

            # Check by class
            footer_by_class = soup.find(
                class_=re.compile(r"footer|site-footer|page-footer", re.I)
            )
            if footer_by_class:
                footer_candidates.append(footer_by_class)

            # Use the most likely candidate
            if footer_candidates:
                footer = footer_candidates[0]

        # Analyze footer if found
        if footer:
            footer_info["present"] = True
            footer_info["semantic"] = footer.name == "footer"
            footer_info["id"] = footer.get("id", "")
            footer_info["classes"] = footer.get("class", [])

            # Analyze content
            footer_info["content"] = {
                "links": len(footer.find_all("a")),
                "navigation": bool(footer.find("nav")),
                "nav_links": self._extract_footer_links(footer),
                "social_links": self._extract_social_links(footer),
                "copyright": self._extract_copyright(footer),
                "has_logo": bool(footer.find("img", alt=re.compile(r"logo", re.I)))
                or bool(footer.find(class_=re.compile(r"logo", re.I))),
                "contact_info": self._extract_contact_info(footer),
                "has_form": bool(footer.find("form")),
                "columns": self._estimate_footer_columns(footer),
            }

        return footer_info

    def _extract_footer_links(self, footer: Tag) -> List[Dict[str, str]]:
        """
        Extract main navigation links from footer.

        Args:
            footer: BeautifulSoup Tag of the footer

        Returns:
            List of link information
        """
        # First check if there's a nav element
        nav = footer.find("nav")
        if nav:
            links = nav.find_all("a")
        else:
            # If no nav, get all links
            links = footer.find_all("a")

        # Filter social links
        nav_links = []
        for link in links:
            # Skip likely social links
            if self._is_social_media_link(link.get("href", "")):
                continue

            # Skip images-only links
            if link.find("img") and not link.get_text().strip():
                continue

            link_text = link.get_text().strip()
            if link_text:
                nav_links.append({"text": link_text, "url": link.get("href", "")})

        return nav_links[:10]  # Return top 10 links

    def _extract_social_links(self, footer: Tag) -> List[Dict[str, str]]:
        """
        Extract social media links from footer.

        Args:
            footer: BeautifulSoup Tag of the footer

        Returns:
            List of social link information
        """
        social_links = []

        # Look for links with social media patterns
        for link in footer.find_all("a", href=True):
            href = link.get("href", "")

            if self._is_social_media_link(href):
                platform = self._identify_social_platform(href)

                social_links.append(
                    {
                        "platform": platform,
                        "url": href,
                        "text": link.get_text().strip(),
                        "has_icon": bool(link.find(["i", "svg", "img"])),
                    }
                )

        return social_links

    def _is_social_media_link(self, url: str) -> bool:
        """
        Check if a URL is a social media link.

        Args:
            url: URL to check

        Returns:
            Boolean indicating if URL is a social media link
        """
        if not url:
            return False

        social_patterns = [
            r"facebook\.com",
            r"twitter\.com",
            r"instagram\.com",
            r"linkedin\.com",
            r"youtube\.com",
            r"pinterest\.com",
            r"tiktok\.com",
            r"snapchat\.com",
            r"t\.me",
            r"reddit\.com",
            r"github\.com",
            r"medium\.com",
            r"vk\.com",
            r"weibo\.com",
            r"tumblr\.com",
            r"flickr\.com",
            r"whatsapp\.com",
            r"telegram\.org",
            r"discord\.gg",
        ]

        return any(re.search(pattern, url, re.I) for pattern in social_patterns)

    def _identify_social_platform(self, url: str) -> str:
        """
        Identify social media platform from URL.

        Args:
            url: Social media URL

        Returns:
            Platform name or 'unknown'
        """
        platforms = {
            r"facebook\.com": "facebook",
            r"twitter\.com": "twitter",
            r"instagram\.com": "instagram",
            r"linkedin\.com": "linkedin",
            r"youtube\.com": "youtube",
            r"pinterest\.com": "pinterest",
            r"tiktok\.com": "tiktok",
            r"snapchat\.com": "snapchat",
            r"t\.me": "telegram",
            r"reddit\.com": "reddit",
            r"github\.com": "github",
            r"medium\.com": "medium",
            r"vk\.com": "vk",
            r"weibo\.com": "weibo",
            r"tumblr\.com": "tumblr",
            r"flickr\.com": "flickr",
            r"whatsapp\.com": "whatsapp",
            r"telegram\.org": "telegram",
            r"discord\.gg": "discord",
        }

        for pattern, platform in platforms.items():
            if re.search(pattern, url, re.I):
                return platform

        return "unknown"

    def _extract_copyright(self, footer: Tag) -> str:
        """
        Extract copyright information from footer.

        Args:
            footer: BeautifulSoup Tag of the footer

        Returns:
            Copyright text or empty string
        """
        # Look for copyright symbol
        text = footer.get_text()
        copyright_match = re.search(r"©|\(c\)|\(C\)|\bCopyright\b.*?\d{4}", text)

        if copyright_match:
            # Extract sentence containing copyright
            start = copyright_match.start()
            end = text.find(".", start)
            if end == -1:
                end = len(text)

            return text[start:end].strip()

        return ""

    def _extract_contact_info(self, footer: Tag) -> Dict[str, str]:
        """
        Extract contact information from footer.

        Args:
            footer: BeautifulSoup Tag of the footer

        Returns:
            Dictionary with contact information
        """
        text = footer.get_text()

        # Extract email
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        email = email_match.group(0) if email_match else ""

        # Extract phone
        phone_match = re.search(
            r"(?:\+\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}", text
        )
        phone = phone_match.group(0) if phone_match else ""

        # Extract address (simplified approach)
        address = ""
        address_elem = footer.find(
            lambda tag: tag.name == "p"
            and re.search(
                r"\d+[\s\w]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\.?",
                tag.get_text(),
                re.I,
            )
        )
        if address_elem:
            address = address_elem.get_text().strip()

        return {"email": email, "phone": phone, "address": address}

    def _estimate_footer_columns(self, footer: Tag) -> int:
        """
        Estimate the number of columns in the footer.

        Args:
            footer: BeautifulSoup Tag of the footer

        Returns:
            Estimated number of columns
        """
        # Check for explicit grid system
        if footer.select('.row > .col, .row > [class*="col-"], .columns > .column'):
            columns = footer.select(
                '.row > .col, .row > [class*="col-"], .columns > .column'
            )
            return len(columns)

        # Check for direct children with headings
        column_candidates = []
        for child in footer.find_all(recursive=False):
            if child.find(["h1", "h2", "h3", "h4", "h5", "h6"]):
                column_candidates.append(child)

        if column_candidates:
            return len(column_candidates)

        # Check for lists (common footer pattern)
        lists = footer.find_all(["ul", "ol"], recursive=False)
        if lists and len(lists) > 1:
            return len(lists)

        # Default to 1 column if no pattern detected
        return 1

    def _identify_header(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Identify and analyze header section.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with header information
        """
        header_info = {
            "present": False,
            "semantic": False,
            "id": "",
            "classes": [],
            "content": {},
        }

        # Look for semantic header
        header = soup.find("header")

        # If not found, look for common header patterns
        if not header:
            header_candidates = []

            # Check by ID
            header_by_id = soup.find(id=re.compile(r"header", re.I))
            if header_by_id:
                header_candidates.append(header_by_id)

            # Check by class
            header_by_class = soup.find(
                class_=re.compile(r"header|site-header|page-header", re.I)
            )
            if header_by_class:
                header_candidates.append(header_by_class)

            # Use the most likely candidate
            if header_candidates:
                header = header_candidates[0]

        # Analyze header if found
        if header:
            header_info["present"] = True
            header_info["semantic"] = header.name == "header"
            header_info["id"] = header.get("id", "")
            header_info["classes"] = header.get("class", [])

            # Analyze content
            header_info["content"] = {
                "has_logo": bool(header.find("img", alt=re.compile(r"logo", re.I)))
                or bool(header.find(class_=re.compile(r"logo", re.I))),
                "has_navigation": bool(header.find("nav"))
                or bool(header.find(class_=re.compile(r"menu|nav", re.I))),
                "has_search": bool(header.find("form", id=re.compile(r"search", re.I)))
                or bool(header.find("input", attrs={"type": "search"})),
                "has_cta": bool(
                    header.find("a", class_=re.compile(r"btn|button|cta", re.I))
                ),
                "navigation_type": self._determine_navigation_type(header),
                "sticky": self._is_sticky_header(header),
            }

        return header_info

    def _determine_navigation_type(self, header: Tag) -> str:
        """
        Determine the type of navigation in header.

        Args:
            header: BeautifulSoup Tag of the header

        Returns:
            Navigation type string
        """
        # Check for hamburger menu pattern (mobile)
        hamburger = header.find(
            class_=re.compile(r"hamburger|toggle|navbar-toggler", re.I)
        )
        if hamburger:
            return "responsive"

        # Check for standard navigation
        nav = header.find("nav")
        if nav and nav.find("ul") and len(nav.find_all("li")) > 3:
            return "standard"

        # Check for mega menu
        mega_menu = header.find(class_=re.compile(r"mega-menu|megamenu", re.I))
        if mega_menu:
            return "mega_menu"

        # Check for dropdown menu
        dropdown = header.select("ul ul") or header.find(
            class_=re.compile(r"dropdown|submenu", re.I)
        )
        if dropdown:
            return "dropdown"

        return "simple"

    def _is_sticky_header(self, header: Tag) -> bool:
        """
        Determine if header is likely sticky/fixed.

        Args:
            header: BeautifulSoup Tag of the header

        Returns:
            Boolean indicating if header is likely sticky
        """
        if header.get("style"):
            style = header.get("style").lower()
            if (
                "position:fixed" in style
                or "position: fixed" in style
                or "position:sticky" in style
                or "position: sticky" in style
            ):
                return True

        sticky_classes = ["sticky", "fixed", "fixed-top", "navbar-fixed"]
        if header.get("class"):
            for cls in header.get("class"):
                if any(sticky_class in cls.lower() for sticky_class in sticky_classes):
                    return True

        return False

    def _analyze_lists(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze lists on the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with list analysis
        """
        unordered_lists = soup.find_all("ul")
        ordered_lists = soup.find_all("ol")

        # Filter out likely navigation and menu lists
        content_uls = [ul for ul in unordered_lists if not self._is_navigation_list(ul)]
        content_ols = ordered_lists

        # Calculate average items per list
        ul_items = sum(len(ul.find_all("li", recursive=False)) for ul in content_uls)
        ol_items = sum(len(ol.find_all("li", recursive=False)) for ol in content_ols)

        avg_ul_items = ul_items / max(len(content_uls), 1)
        avg_ol_items = ol_items / max(len(content_ols), 1)

        # Check for nested lists
        nested_lists = sum(
            1 for lst in unordered_lists + ordered_lists if lst.find(["ul", "ol"])
        )

        # Check for special list types
        definition_lists = soup.find_all("dl")

        return {
            "unordered_lists": len(unordered_lists),
            "ordered_lists": len(ordered_lists),
            "content_lists": len(content_uls) + len(content_ols),
            "navigation_lists": len(unordered_lists) - len(content_uls),
            "definition_lists": len(definition_lists),
            "nested_lists": nested_lists,
            "avg_ul_items": avg_ul_items,
            "avg_ol_items": avg_ol_items,
            "total_list_items": ul_items + ol_items,
        }

    def _is_navigation_list(self, ul: Tag) -> bool:
        """
        Determine if an unordered list is likely navigation.

        Args:
            ul: BeautifulSoup Tag of unordered list

        Returns:
            Boolean indicating if list is likely navigation
        """
        # Check if list is inside navigation
        if ul.parent.name == "nav" or any(
            "nav" in cls.lower() for cls in ul.parent.get("class", [])
        ):
            return True

        # Check list classes
        if ul.get("class") and any(
            cls.lower() in ["menu", "nav", "navigation", "navbar"]
            for cls in ul.get("class")
        ):
            return True

        # Check if all list items contain links
        list_items = ul.find_all("li", recursive=False)
        if list_items and all(bool(li.find("a")) for li in list_items):
            # Not all link lists are navigation, check additional factors
            links = [li.find("a") for li in list_items]
            link_texts = [link.get_text().strip() for link in links if link]

            # If link texts are all short, likely a menu
            if all(len(text) < 20 for text in link_texts if text):
                return True

        return False

    def _analyze_tables(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze tables on the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with table analysis
        """
        tables = soup.find_all("table")

        if not tables:
            return {
                "count": 0,
                "uses_thead": 0,
                "uses_th": 0,
                "uses_caption": 0,
                "avg_rows": 0,
                "avg_cols": 0,
                "has_responsive_tables": False,
            }

        # Analyze table features
        uses_thead = sum(1 for table in tables if table.find("thead"))
        uses_th = sum(1 for table in tables if table.find("th"))
        uses_caption = sum(1 for table in tables if table.find("caption"))

        # Calculate average dimensions
        rows_per_table = []
        cols_per_table = []

        for table in tables:
            rows = table.find_all("tr")
            rows_per_table.append(len(rows))

            if rows:
                # Get column count from first row
                first_row = rows[0]
                cols = first_row.find_all(["td", "th"])
                cols_per_table.append(len(cols))

        avg_rows = sum(rows_per_table) / len(tables)
        avg_cols = sum(cols_per_table) / len(tables) if cols_per_table else 0

        # Check for responsive tables
        responsive_containers = soup.find_all(
            class_=re.compile(r"table-responsive|responsive-table", re.I)
        )
        responsive_tables = len(responsive_containers) > 0 or any(
            "overflow" in table.get("style", "") for table in tables
        )

        return {
            "count": len(tables),
            "uses_thead": uses_thead,
            "uses_th": uses_th,
            "uses_caption": uses_caption,
            "avg_rows": avg_rows,
            "avg_cols": avg_cols,
            "has_responsive_tables": responsive_tables,
            "layout_tables": sum(
                1 for table in tables if not table.find(["th", "thead"])
            ),
        }

    def _analyze_iframes(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze iframe usage on the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with iframe analysis
        """
        iframes = soup.find_all("iframe")

        if not iframes:
            return {"count": 0, "types": {}}

        # Categorize iframes
        iframe_types = {"map": 0, "video": 0, "form": 0, "social": 0, "other": 0}

        for iframe in iframes:
            src = iframe.get("src", "").lower()

            # Categorize based on source
            if re.search(r"(google|yandex|bing|osm).*map", src):
                iframe_types["map"] += 1
            elif re.search(r"(youtube|vimeo|dailymotion|twitch)", src):
                iframe_types["video"] += 1
            elif re.search(r"(typeform|google.*forms|wufoo)", src):
                iframe_types["form"] += 1
            elif re.search(r"(facebook|twitter|instagram|linkedin)", src):
                iframe_types["social"] += 1
            else:
                iframe_types["other"] += 1

        return {
            "count": len(iframes),
            "types": iframe_types,
            "srcs": [
                iframe.get("src", "") for iframe in iframes[:5]
            ],  # List first 5 iframes
        }

    def _analyze_scripts(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze script usage on the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with script analysis
        """
        scripts = soup.find_all("script")

        # Count script types
        inline_scripts = sum(1 for script in scripts if not script.get("src"))
        external_scripts = sum(1 for script in scripts if script.get("src"))

        # Detect common libraries and frameworks
        script_srcs = [script.get("src", "") for script in scripts if script.get("src")]

        libraries_detected = {
            "jquery": any(re.search(r"jquery", src, re.I) for src in script_srcs),
            "bootstrap": any(re.search(r"bootstrap", src, re.I) for src in script_srcs),
            "vue": any(re.search(r"vue", src, re.I) for src in script_srcs),
            "react": any(re.search(r"react", src, re.I) for src in script_srcs),
            "angular": any(re.search(r"angular", src, re.I) for src in script_srcs),
            "gsap": any(re.search(r"gsap", src, re.I) for src in script_srcs),
            "lodash": any(re.search(r"lodash", src, re.I) for src in script_srcs),
            "moment": any(re.search(r"moment", src, re.I) for src in script_srcs),
            "analytics": any(
                re.search(r"(google.*analytics|gtag|gtm|ga\.js)", src, re.I)
                for src in script_srcs
            ),
        }

        # Check inline scripts for libraries
        inline_script_text = " ".join(
            script.get_text() for script in scripts if not script.get("src")
        )

        if not libraries_detected["jquery"] and re.search(
            r"\$\(|\bjQuery\b", inline_script_text
        ):
            libraries_detected["jquery"] = True

        if not libraries_detected["react"] and re.search(
            r"React\.|\bReactDOM\b", inline_script_text
        ):
            libraries_detected["react"] = True

        if not libraries_detected["vue"] and re.search(
            r"Vue\.|\bVue\(", inline_script_text
        ):
            libraries_detected["vue"] = True

        return {
            "total_count": len(scripts),
            "inline_scripts": inline_scripts,
            "external_scripts": external_scripts,
            "libraries_detected": libraries_detected,
            "defer_scripts": sum(1 for script in scripts if script.get("defer")),
            "async_scripts": sum(1 for script in scripts if script.get("async")),
            "module_scripts": sum(
                1 for script in scripts if script.get("type") == "module"
            ),
        }

    def _analyze_stylesheets(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze stylesheet usage on the page.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with stylesheet analysis
        """
        # External stylesheets
        stylesheets = soup.find_all("link", rel="stylesheet") + soup.find_all(
            "link", rel=lambda x: x and "stylesheet" in x
        )

        # Inline styles
        style_tags = soup.find_all("style")
        inline_styles = sum(1 for elem in soup.find_all(style=True))

        # Detect CSS frameworks
        stylesheet_hrefs = [link.get("href", "") for link in stylesheets]

        frameworks_detected = {
            "bootstrap": any(
                re.search(r"bootstrap", href, re.I) for href in stylesheet_hrefs
            ),
            "foundation": any(
                re.search(r"foundation", href, re.I) for href in stylesheet_hrefs
            ),
            "bulma": any(re.search(r"bulma", href, re.I) for href in stylesheet_hrefs),
            "tailwind": any(
                re.search(r"tailwind", href, re.I) for href in stylesheet_hrefs
            ),
            "materialize": any(
                re.search(r"materialize", href, re.I) for href in stylesheet_hrefs
            ),
            "fontawesome": any(
                re.search(r"(font-awesome|fontawesome)", href, re.I)
                for href in stylesheet_hrefs
            ),
        }

        # Check inline styles for frameworks
        inline_style_text = " ".join(style.get_text() for style in style_tags)

        if not frameworks_detected["bootstrap"] and re.search(
            r"\.navbar-|\.btn-|\.col-", inline_style_text
        ):
            frameworks_detected["bootstrap"] = True

        if not frameworks_detected["tailwind"] and re.search(
            r"\.text-\w+\d+|\.bg-\w+\d+|\.flex\s", inline_style_text
        ):
            frameworks_detected["tailwind"] = True

        return {
            "external_stylesheets": len(stylesheets),
            "style_tags": len(style_tags),
            "inline_styles": inline_styles,
            "frameworks_detected": frameworks_detected,
            "print_stylesheet": any(
                "print" in link.get("media", "") for link in stylesheets
            ),
            "mobile_stylesheet": any(
                "mobile" in link.get("media", "") or "handheld" in link.get("media", "")
                for link in stylesheets
            ),
            "responsive_media_queries": any(
                "@media" in style.get_text()
                and ("max-width" in style.get_text() or "min-width" in style.get_text())
                for style in style_tags
            ),
        }

    def _calculate_complexity(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Calculate overall page complexity metrics.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with complexity metrics
        """
        # Count elements
        total_elements = len(soup.find_all())
        unique_tags = len(set(tag.name for tag in soup.find_all()))
        max_nesting = self._calculate_max_nesting(soup)

        # Count classes and IDs
        elements_with_class = soup.find_all(class_=True)
        elements_with_id = soup.find_all(id=True)

        all_classes = []
        for elem in elements_with_class:
            if isinstance(elem.get("class"), list):
                all_classes.extend(elem.get("class"))
            else:
                all_classes.append(elem.get("class"))

        unique_classes = len(set(all_classes))
        unique_ids = len(set(elem.get("id") for elem in elements_with_id))

        # Calculate text-to-code ratio
        html_length = len(str(soup))
        text_length = len(soup.get_text())
        text_ratio = text_length / html_length if html_length > 0 else 0

        # Determine complexity level
        if total_elements < 300 and max_nesting < 10:
            complexity_level = "simple"
        elif total_elements < 1000 and max_nesting < 15:
            complexity_level = "moderate"
        else:
            complexity_level = "complex"

        return {
            "total_elements": total_elements,
            "unique_tags": unique_tags,
            "max_nesting": max_nesting,
            "elements_with_class": len(elements_with_class),
            "elements_with_id": len(elements_with_id),
            "unique_classes": unique_classes,
            "unique_ids": unique_ids,
            "text_to_code_ratio": text_ratio,
            "complexity_level": complexity_level,
        }

    def _calculate_max_nesting(self, soup: BeautifulSoup) -> int:
        """
        Calculate maximum nesting level of the document.

        Args:
            soup: BeautifulSoup object

        Returns:
            Maximum nesting depth
        """
        max_depth = 0

        for elem in soup.find_all():
            depth = len(list(elem.parents))
            max_depth = max(max_depth, depth)

        return max_depth

    def _determine_element_location(self, elem: Tag) -> str:
        """
        Determine the location of an element in the page.

        Args:
            elem: BeautifulSoup Tag

        Returns:
            Location string ('header', 'footer', 'sidebar', 'main', 'unknown')
        """
        # Check if element is in a semantic section
        for parent in elem.parents:
            if (
                parent.name == "header"
                or ("id" in parent.attrs and "header" in parent["id"].lower())
                or (
                    "class" in parent.attrs
                    and any("header" in cls.lower() for cls in parent["class"])
                )
            ):
                return "header"

            if (
                parent.name == "footer"
                or ("id" in parent.attrs and "footer" in parent["id"].lower())
                or (
                    "class" in parent.attrs
                    and any("footer" in cls.lower() for cls in parent["class"])
                )
            ):
                return "footer"

            if (
                parent.name == "aside"
                or ("id" in parent.attrs and "sidebar" in parent["id"].lower())
                or (
                    "class" in parent.attrs
                    and any("sidebar" in cls.lower() for cls in parent["class"])
                )
            ):
                return "sidebar"

            if (
                parent.name == "main"
                or parent.name == "article"
                or (
                    "id" in parent.attrs
                    and (
                        "main" in parent["id"].lower()
                        or "content" in parent["id"].lower()
                    )
                )
                or (
                    "class" in parent.attrs
                    and any(
                        ("main" in cls.lower() or "content" in cls.lower())
                        for cls in parent["class"]
                    )
                )
            ):
                return "main"

            if (
                parent.name == "nav"
                or ("id" in parent.attrs and "nav" in parent["id"].lower())
                or (
                    "class" in parent.attrs
                    and any("nav" in cls.lower() for cls in parent["class"])
                )
            ):
                return "navigation"

        return "unknown"
