from typing import Dict, Any, List, Optional, Union
import re
from collections import Counter
import asyncio

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import urlparse

from app.services.analyzer.base_analyzer import BaseAnalyzer
from app.services.analyzer.utils.text_utils import TextProcessor


class SEOAnalyzer(BaseAnalyzer):
    """SEO analyzer module for web content analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SEO analyzer with configuration."""
        super().__init__(config)
        self.config = config or {}
        # self.nlp = spacy.load(self.config.get("spacy_model", "en_core_web_sm"))
        self.text_processor = TextProcessor()
        self.stop_words = set(stopwords.words("english"))

        # Default scoring weights
        self.scoring_weights = self.config.get(
            "scoring_weights",
            {
                "meta_tags": 0.20,
                "headings": 0.15,
                "content_relevance": 0.25,
                "url_structure": 0.10,
                "keyword_distribution": 0.20,
                "over_optimization": 0.10,
            },
        )

    async def analyze(
        self,
        processed_text: Dict[str, Any],
        html_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        url: Optional[str] = None,
        structure: Optional[Dict[str, Any]] = None,
        target_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive SEO analysis on the web content.

        Args:
            processed_text: The preprocessed text data
            html_content: The raw HTML content of the page
            metadata: Page metadata including title, description, etc.
            url: The URL of the page
            structure: The HTML structure information
            target_keywords: Optional list of target keywords to analyze against

        Returns:
            Dict containing SEO analysis results and scores
        """
        metadata = metadata or {}
        text = processed_text.get("normalized_text", "")

        # Parse HTML if provided
        soup = BeautifulSoup(html_content, "html.parser") if html_content else None

        # Run analyses in parallel
        meta_analysis, heading_analysis, url_analysis, keyword_analysis = (
            await asyncio.gather(
                self.analyze_meta_tags(soup, metadata, target_keywords),
                self.analyze_headings(soup, text, target_keywords),
                self.analyze_url(url, target_keywords),
                self.analyze_keyword_usage(text, soup, target_keywords),
            )
        )

        # Calculate overall optimization status
        optimization_analysis = self.analyze_optimization_level(
            text, meta_analysis, heading_analysis, keyword_analysis
        )

        # Content relevance analysis
        content_relevance = await self.analyze_content_relevance(
            text, meta_analysis.get("title", {}).get("content", ""), target_keywords
        )

        # LSI keywords analysis
        lsi_analysis = await self.analyze_lsi_keywords(text, target_keywords)

        # Combine all analyses and calculate overall score
        combined_analysis = {
            "meta_tags": meta_analysis,
            "headings": heading_analysis,
            "url_structure": url_analysis,
            "keyword_usage": keyword_analysis,
            "optimization_level": optimization_analysis,
            "content_relevance": content_relevance,
            "lsi_keywords": lsi_analysis,
            "recommendations": self.generate_recommendations(
                meta_analysis,
                heading_analysis,
                url_analysis,
                keyword_analysis,
                optimization_analysis,
                content_relevance,
                lsi_analysis,
            ),
        }

        # Calculate the overall SEO score
        combined_analysis["overall_score"] = self.calculate_overall_score(
            combined_analysis
        )

        return combined_analysis

    async def analyze_meta_tags(
        self,
        soup: BeautifulSoup,
        metadata: Dict[str, Any],
        target_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze meta tags for SEO optimization.

        Args:
            soup: BeautifulSoup object of the page HTML
            metadata: Extracted metadata from the page
            target_keywords: Optional list of target keywords

        Returns:
            Dict with meta tag analysis results
        """
        results = {
            "title": {
                "content": metadata.get("title", ""),
                "length": len(metadata.get("title", "")),
                "issues": [],
            },
            "description": {
                "content": metadata.get("description", ""),
                "length": len(metadata.get("description", "")),
                "issues": [],
            },
            "keywords": {
                "content": metadata.get("keywords", []),
                "count": len(metadata.get("keywords", [])),
                "issues": [],
            },
            "score": 0.0,
            "issues": [],
            "recommendations": [],
        }

        # Title analysis
        title = results["title"]["content"]
        if not title:
            results["title"]["issues"].append("Missing title tag")
            results["issues"].append("Missing title tag")
            results["recommendations"].append(
                "Add a descriptive title tag (recommended length: 50-60 characters)"
            )
        elif results["title"]["length"] < 30:
            results["title"]["issues"].append("Title tag is too short")
            results["issues"].append("Title tag is too short")
            results["recommendations"].append(
                "Increase title length to 50-60 characters"
            )
        elif results["title"]["length"] > 60:
            results["title"]["issues"].append(
                "Title tag is too long and may be truncated in search results"
            )
            results["issues"].append("Title tag is too long")
            results["recommendations"].append(
                "Reduce title length to 50-60 characters to prevent truncation in search results"
            )

        # Check if title contains target keywords
        if target_keywords and title:
            title_lower = title.lower()
            keywords_in_title = [
                kw for kw in target_keywords if kw.lower() in title_lower
            ]
            if not keywords_in_title:
                results["title"]["issues"].append(
                    "Title does not contain any target keywords"
                )
                results["issues"].append("Title missing target keywords")
                results["recommendations"].append(
                    f"Include primary keywords in the title: {', '.join(target_keywords[:3])}"
                )
            results["title"]["keywords_included"] = keywords_in_title

        # Description analysis
        description = results["description"]["content"]
        if not description:
            results["description"]["issues"].append("Missing meta description")
            results["issues"].append("Missing meta description")
            results["recommendations"].append(
                "Add a meta description (recommended length: 150-160 characters)"
            )
        elif results["description"]["length"] < 70:
            results["description"]["issues"].append("Meta description is too short")
            results["issues"].append("Meta description is too short")
            results["recommendations"].append(
                "Increase meta description length to 150-160 characters"
            )
        elif results["description"]["length"] > 160:
            results["description"]["issues"].append(
                "Meta description is too long and may be truncated in search results"
            )
            results["issues"].append("Meta description is too long")
            results["recommendations"].append(
                "Reduce meta description length to 150-160 characters to prevent truncation"
            )

        # Check if description contains target keywords
        if target_keywords and description:
            desc_lower = description.lower()
            keywords_in_desc = [
                kw for kw in target_keywords if kw.lower() in desc_lower
            ]
            if not keywords_in_desc:
                results["description"]["issues"].append(
                    "Description does not contain any target keywords"
                )
                results["issues"].append("Description missing target keywords")
                results["recommendations"].append(
                    f"Include primary keywords in the description: {', '.join(target_keywords[:3])}"
                )
            results["description"]["keywords_included"] = keywords_in_desc

        # Calculate score based on issues
        total_issues = len(results["issues"])
        if total_issues == 0:
            results["score"] = 1.0
        else:
            results["score"] = max(0.0, 1.0 - (total_issues * 0.2))

        return results

    async def analyze_headings(
        self,
        soup: BeautifulSoup,
        text: str,
        target_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze heading structure and content for SEO.

        Args:
            soup: BeautifulSoup object of the page HTML
            text: Extracted text content
            target_keywords: Optional list of target keywords

        Returns:
            Dict with heading analysis results
        """
        results = {
            "headings": {},
            "hierarchy": {"is_valid": True, "issues": []},
            "keyword_usage": {},
            "score": 0.0,
            "issues": [],
            "recommendations": [],
        }

        # Extract all headings
        heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
        headings = {}
        previous_level = 0

        for tag in heading_tags:
            heading_elements = soup.find_all(tag)
            headings[tag] = [h.get_text().strip() for h in heading_elements]

            level = int(tag[1])

            # Check for skipped heading levels (e.g., h1 -> h3 without h2)
            if level - previous_level > 1 and previous_level > 0 and heading_elements:
                results["hierarchy"]["is_valid"] = False
                results["hierarchy"]["issues"].append(
                    f"Skipped heading level: {previous_level} to {level}"
                )
                results["issues"].append(
                    f"Improper heading hierarchy: h{previous_level} to h{level}"
                )
                results["recommendations"].append(
                    f"Fix heading hierarchy: use h{previous_level+1} instead of h{level} after h{previous_level}"
                )

            if heading_elements:
                previous_level = level

        results["headings"] = headings

        # H1 analysis
        if not headings.get("h1"):
            results["issues"].append("Missing H1 heading")
            results["recommendations"].append(
                "Add an H1 heading containing your primary keyword"
            )
        elif len(headings.get("h1", [])) > 1:
            results["issues"].append(
                "Multiple H1 headings (recommended: only one H1 per page)"
            )
            results["recommendations"].append("Use only one H1 heading per page")

        # Keyword usage in headings
        if target_keywords:
            keyword_in_headings = {kw: [] for kw in target_keywords}

            for tag, heading_list in headings.items():
                for i, heading in enumerate(heading_list):
                    heading_lower = heading.lower()
                    for kw in target_keywords:
                        if kw.lower() in heading_lower:
                            keyword_in_headings[kw].append(f"{tag}_{i+1}")

            results["keyword_usage"] = keyword_in_headings

            # Check if primary keywords are used in headings
            primary_kw = target_keywords[0] if target_keywords else None
            if primary_kw and not any(keyword_in_headings.get(primary_kw, [])):
                results["issues"].append(
                    f"Primary keyword '{primary_kw}' not found in any heading"
                )
                results["recommendations"].append(
                    f"Include primary keyword '{primary_kw}' in H1 and/or H2 headings"
                )

        # Calculate score based on issues
        total_issues = len(results["issues"])
        if total_issues == 0:
            results["score"] = 1.0
        else:
            results["score"] = max(0.0, 1.0 - (total_issues * 0.2))

        return results

    async def analyze_url(
        self, url: str, target_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze URL structure for SEO factors.

        Args:
            url: Page URL
            target_keywords: Optional list of target keywords

        Returns:
            Dict with URL analysis results
        """
        parsed_url = urlparse(url)
        path = parsed_url.path

        results = {
            "url": url,
            "path": path,
            "length": len(path),
            "segments": [],
            "keyword_usage": {},
            "issues": [],
            "recommendations": [],
            "score": 0.0,
        }

        # Split the path into segments
        segments = [s for s in path.split("/") if s]
        results["segments"] = segments
        results["segment_count"] = len(segments)

        # Check for common URL issues
        if results["length"] > 100:
            results["issues"].append("URL is too long (over 100 characters)")
            results["recommendations"].append(
                "Shorten the URL to less than 100 characters"
            )

        # Check for URL slugs that are SEO-friendly
        non_seo_friendly = []
        for segment in segments:
            # Check for numbers, special chars, uppercase
            if re.search(r"[A-Z]", segment):
                non_seo_friendly.append(f"'{segment}' contains uppercase characters")
            if re.search(r"[^a-zA-Z0-9-]", segment):
                non_seo_friendly.append(
                    f"'{segment}' contains special characters other than hyphens"
                )
            if len(segment) > 30:
                non_seo_friendly.append(f"'{segment}' is very long")

        if non_seo_friendly:
            results["issues"].extend(non_seo_friendly)
            results["recommendations"].append(
                "Make URL segments lowercase, use hyphens instead of underscores or spaces, avoid special characters"
            )

        # Check if URL contains target keywords
        if target_keywords:
            keyword_presence = {
                kw: kw.lower() in path.lower() for kw in target_keywords
            }
            results["keyword_usage"] = keyword_presence

            if not any(keyword_presence.values()):
                results["issues"].append("URL does not contain any target keywords")
                results["recommendations"].append(
                    f"Include primary keywords in the URL: {', '.join(target_keywords[:2])}"
                )

        # Calculate score based on issues
        total_issues = len(results["issues"])
        if total_issues == 0:
            results["score"] = 1.0
        else:
            results["score"] = max(0.0, 1.0 - (total_issues * 0.15))

        return results

    async def analyze_keyword_usage(
        self,
        text: str,
        soup: BeautifulSoup,
        target_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze keyword usage, density, and distribution in content.

        Args:
            text: Extracted text content
            soup: BeautifulSoup object of the page HTML
            target_keywords: Optional list of target keywords

        Returns:
            Dict with keyword usage analysis results
        """
        if not text:
            return {
                "keywords": {},
                "density": {},
                "distribution": {},
                "issues": ["No text content found to analyze"],
                "recommendations": ["Add substantial text content to the page"],
                "score": 0.0,
            }

        # Tokenize and clean text
        words = word_tokenize(text.lower())
        words_no_stop = [w for w in words if w.isalpha() and w not in self.stop_words]

        word_count = len(words_no_stop)

        results = {
            "word_count": word_count,
            "keywords": {},
            "density": {},
            "distribution": {},
            "issues": [],
            "recommendations": [],
            "score": 0.0,
        }

        if word_count == 0:
            results["issues"].append("No meaningful text content found")
            results["recommendations"].append(
                "Add substantial text content to the page"
            )
            results["score"] = 0.0
            return results

        # If no target keywords provided, extract them using TF-IDF
        if not target_keywords:
            tfidf_vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
            tfidf_matrix = tfidf_vectorizer.fit_transform([text])
            feature_names = tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Get top keywords by TF-IDF score
            target_keywords = [
                feature_names[i] for i in tfidf_scores.argsort()[-10:][::-1]
            ]

        # Analyze keyword usage
        keyword_counts = {}
        keyword_density = {}

        for keyword in target_keywords:
            # Handle multi-word keywords
            if " " in keyword:
                keyword_lower = keyword.lower()
                count = text.lower().count(keyword_lower)
            else:
                keyword_lower = keyword.lower()
                count = sum(1 for word in words if word.lower() == keyword_lower)

            keyword_counts[keyword] = count

            # Calculate density (as percentage)
            if word_count > 0:
                density = (count / word_count) * 100
                keyword_density[keyword] = round(density, 2)
            else:
                keyword_density[keyword] = 0.0

        results["keywords"] = keyword_counts
        results["density"] = keyword_density

        # Check for keyword density issues
        density_issues = []
        for keyword, density in keyword_density.items():
            if density == 0:
                density_issues.append(f"Keyword '{keyword}' not found in content")
            elif density < 0.5:
                density_issues.append(
                    f"Keyword '{keyword}' density is low ({density}%)"
                )
            elif density > 3.0:
                density_issues.append(
                    f"Keyword '{keyword}' density is too high ({density}%)"
                )

        if density_issues:
            results["issues"].extend(density_issues)
            results["recommendations"].append(
                "Maintain keyword density between 0.5% and 3% for primary keywords"
            )

        # Analyze distribution through content (by dividing content into sections)
        if word_count >= 300:
            sections = self.divide_text_into_sections(text)
            distribution = {}

            for keyword in target_keywords:
                keyword_lower = keyword.lower()
                section_counts = []

                for section in sections:
                    if " " in keyword:
                        count = section.lower().count(keyword_lower)
                    else:
                        section_words = word_tokenize(section.lower())
                        count = sum(
                            1 for word in section_words if word == keyword_lower
                        )

                    section_counts.append(count)

                distribution[keyword] = section_counts

            results["distribution"] = distribution

            # Check if keywords are well-distributed
            for keyword, counts in distribution.items():
                if sum(counts) > 0 and any(count == 0 for count in counts):
                    results["issues"].append(
                        f"Keyword '{keyword}' is not well distributed throughout the content"
                    )
                    results["recommendations"].append(
                        f"Distribute keyword '{keyword}' more evenly throughout the content"
                    )

        # Calculate score based on issues
        total_issues = len(results["issues"])
        if total_issues == 0:
            results["score"] = 1.0
        else:
            results["score"] = max(0.0, 1.0 - (total_issues * 0.1))

        return results

    def analyze_optimization_level(
        self,
        text: str,
        meta_analysis: Dict[str, Any],
        heading_analysis: Dict[str, Any],
        keyword_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Determine if content is under-optimized, well-optimized, or over-optimized.

        Args:
            text: Extracted text content
            meta_analysis: Results from meta tags analysis
            heading_analysis: Results from headings analysis
            keyword_analysis: Results from keyword analysis

        Returns:
            Dict with optimization level analysis
        """
        results = {"status": "", "score": 0.0, "issues": [], "recommendations": []}

        # Check for over-optimization signals
        over_optimization = []

        # Check keyword stuffing in meta tags
        if meta_analysis and meta_analysis.get("title", {}).get("content"):
            title = meta_analysis["title"]["content"]
            if title and self.is_keyword_stuffed(title):
                over_optimization.append("Title tag appears to be keyword stuffed")

        # Check keyword stuffing in content
        if text and self.is_keyword_stuffed(text):
            over_optimization.append("Content appears to be keyword stuffed")

        # Check excessive keyword density
        if keyword_analysis and keyword_analysis.get("density"):
            for keyword, density in keyword_analysis["density"].items():
                if density > 4.0:
                    over_optimization.append(
                        f"Excessive keyword density for '{keyword}' ({density}%)"
                    )

        # Check for under-optimization
        under_optimization = []

        # Missing meta tags
        if meta_analysis and "Missing title tag" in meta_analysis.get("issues", []):
            under_optimization.append("Missing title tag")

        if meta_analysis and "Missing meta description" in meta_analysis.get(
            "issues", []
        ):
            under_optimization.append("Missing meta description")

        # Missing headings
        if heading_analysis and "Missing H1 heading" in heading_analysis.get(
            "issues", []
        ):
            under_optimization.append("Missing H1 heading")

        # Missing keywords
        if keyword_analysis and keyword_analysis.get("keywords"):
            zero_keywords = [
                k for k, v in keyword_analysis["keywords"].items() if v == 0
            ]
            if zero_keywords:
                under_optimization.append(
                    f"Keywords not found in content: {', '.join(zero_keywords[:3])}"
                )

        # Determine optimization status
        if over_optimization:
            results["status"] = "over-optimized"
            results["issues"] = over_optimization
            results["recommendations"] = [
                "Reduce keyword repetition and focus on natural, user-friendly content",
                "Aim for a more diverse vocabulary and LSI keywords",
                "Maintain keyword density below 3% for any keyword",
            ]
            results["score"] = 0.5  # Penalize over-optimization
        elif under_optimization:
            results["status"] = "under-optimized"
            results["issues"] = under_optimization
            results["recommendations"] = [
                "Add missing meta tags and heading elements",
                "Include target keywords in important elements (title, headings, content)",
                "Expand content to fully cover the topic",
            ]
            results["score"] = 0.5  # Penalize under-optimization
        else:
            results["status"] = "well-optimized"
            results["issues"] = []
            results["recommendations"] = [
                "Continue maintaining a good balance of keyword usage",
                "Regularly update content to keep it fresh and relevant",
            ]
            results["score"] = 1.0

        return results

    async def analyze_content_relevance(
        self, text: str, title: str, target_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze content relevance to search intent and target keywords.

        Args:
            text: Extracted text content
            title: Page title
            target_keywords: Optional list of target keywords

        Returns:
            Dict with content relevance analysis results
        """
        results = {
            "topical_relevance": 0.0,
            "semantic_coverage": {},
            "content_depth": "",
            "issues": [],
            "recommendations": [],
            "score": 0.0,
        }

        if not text:
            results["issues"].append("No text content to analyze")
            results["recommendations"].append(
                "Add substantial text content to the page"
            )
            results["score"] = 0.0
            return results

        # Analyze content length
        word_count = len(text.split())
        results["word_count"] = word_count

        if word_count < 300:
            results["content_depth"] = "thin"
            results["issues"].append("Content is too thin (less than 300 words)")
            results["recommendations"].append(
                "Expand content to at least 500-1000 words for better topical coverage"
            )
        elif word_count < 700:
            results["content_depth"] = "moderate"
        else:
            results["content_depth"] = "in-depth"

        # Calculate semantic similarity between content and target keywords
        if target_keywords:
            # Use spaCy for semantic similarity
            doc_text = self.nlp(text[:25000])  # Limit to avoid context length issues

            semantic_coverage = {}
            for keyword in target_keywords:
                doc_keyword = self.nlp(keyword)
                similarity = doc_text.similarity(doc_keyword)
                semantic_coverage[keyword] = round(similarity, 2)

            results["semantic_coverage"] = semantic_coverage

            # Calculate average topical relevance
            if semantic_coverage:
                average_relevance = sum(semantic_coverage.values()) / len(
                    semantic_coverage
                )
                results["topical_relevance"] = round(average_relevance, 2)

                if average_relevance < 0.5:
                    results["issues"].append(
                        "Content has low semantic relevance to target keywords"
                    )
                    results["recommendations"].append(
                        "Improve content relevance by covering topics more directly related to target keywords"
                    )

        # Analyze topic coverage completeness using TF-IDF to extract key phrases
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), max_features=10, stop_words="english"
        )
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform([text])
            feature_names = tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Get top phrases by TF-IDF score
            key_phrases = [feature_names[i] for i in tfidf_scores.argsort()[-10:][::-1]]
            results["key_phrases"] = key_phrases

            # If we have target keywords, check coverage of related concepts
            if target_keywords:
                covered_topics = set()
                for phrase in key_phrases:
                    for keyword in target_keywords:
                        # Check if the key phrase is related to any target keyword
                        if keyword.lower() in phrase.lower() or any(
                            word in phrase.lower() for word in keyword.lower().split()
                        ):
                            covered_topics.add(keyword)

                coverage_ratio = (
                    len(covered_topics) / len(target_keywords) if target_keywords else 0
                )
                results["topic_coverage_ratio"] = round(coverage_ratio, 2)

                if coverage_ratio < 0.7:
                    results["issues"].append(
                        "Content does not fully cover the target topics"
                    )
                    results["recommendations"].append(
                        "Expand content to cover more aspects of the target topics"
                    )
        except:
            # Handle cases where TF-IDF fails (e.g., very short texts)
            results["key_phrases"] = []
            results["topic_coverage_ratio"] = 0.0

        # Calculate score based on issues
        total_issues = len(results["issues"])
        if total_issues == 0:
            results["score"] = 1.0
        else:
            results["score"] = max(0.0, 1.0 - (total_issues * 0.25))

        return results

    async def analyze_lsi_keywords(
        self, text: str, target_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze latent semantic indexing (LSI) keywords in the content.

        Args:
            text: Extracted text content
            target_keywords: Optional list of target keywords

        Returns:
            Dict with LSI keyword analysis results
        """
        results = {
            "lsi_keywords": [],
            "presence": {},
            "coverage": 0.0,
            "issues": [],
            "recommendations": [],
            "score": 0.0,
        }

        if not text or not target_keywords:
            results["issues"].append("Insufficient data for LSI analysis")
            results["recommendations"].append(
                "Provide target keywords and content for LSI analysis"
            )
            results["score"] = 0.0
            return results

        # Extract potential LSI keywords using TF-IDF to extract n-grams
        try:
            tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 3), max_features=20, stop_words="english"
            )
            tfidf_matrix = tfidf_vectorizer.fit_transform([text])
            feature_names = tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Get top phrases by TF-IDF score
            candidates = [feature_names[i] for i in tfidf_scores.argsort()[-20:][::-1]]

            # Filter out exact matches with target keywords
            lsi_keywords = [
                candidate
                for candidate in candidates
                if candidate not in target_keywords
                and all(candidate.lower() != kw.lower() for kw in target_keywords)
            ]

            results["lsi_keywords"] = lsi_keywords[:10]  # Top 10 LSI keywords

            # Check presence of LSI keywords in content
            presence = {}
            for keyword in results["lsi_keywords"]:
                keyword_lower = keyword.lower()
                presence[keyword] = keyword_lower in text.lower()

            results["presence"] = presence

            # Calculate coverage
            if presence:
                coverage = sum(1 for v in presence.values() if v) / len(presence)
                results["coverage"] = round(coverage, 2)

                if coverage < 0.5:
                    results["issues"].append(
                        "Content has low coverage of semantically related terms"
                    )
                    results["recommendations"].append(
                        "Include more semantically related terms for better topical relevance"
                    )
        except:
            results["issues"].append(
                "LSI analysis failed, possibly due to insufficient content"
            )
            results["recommendations"].append(
                "Add more substantial content for better semantic analysis"
            )

        # Calculate score based on issues
        total_issues = len(results["issues"])
        if total_issues == 0:
            results["score"] = 1.0
        else:
            results["score"] = max(0.0, 1.0 - (total_issues * 0.2))

        return results

    def calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate the overall SEO score based on component scores.

        Args:
            analysis: Complete analysis results

        Returns:
            Float representing the overall SEO score (0.0-1.0)
        """
        component_scores = {
            "meta_tags": analysis.get("meta_tags", {}).get("score", 0.0)
            * self.scoring_weights.get("meta_tags", 0.2),
            "headings": analysis.get("headings", {}).get("score", 0.0)
            * self.scoring_weights.get("headings", 0.15),
            "url_structure": analysis.get("url_structure", {}).get("score", 0.0)
            * self.scoring_weights.get("url_structure", 0.1),
            "keyword_usage": analysis.get("keyword_usage", {}).get("score", 0.0)
            * self.scoring_weights.get("keyword_distribution", 0.2),
            "optimization_level": analysis.get("optimization_level", {}).get(
                "score", 0.0
            )
            * self.scoring_weights.get("over_optimization", 0.1),
            "content_relevance": analysis.get("content_relevance", {}).get("score", 0.0)
            * self.scoring_weights.get("content_relevance", 0.25),
        }

        # Sum up the weighted scores
        overall_score = sum(component_scores.values())

        # Round to 2 decimal places
        return round(overall_score, 2)

    def generate_recommendations(
        self,
        meta_analysis: Dict[str, Any],
        heading_analysis: Dict[str, Any],
        url_analysis: Dict[str, Any],
        keyword_analysis: Dict[str, Any],
        optimization_analysis: Dict[str, Any],
        content_relevance: Dict[str, Any],
        lsi_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable SEO recommendations based on analysis results.

        Args:
            Various analysis component results

        Returns:
            List of recommendation objects with priority and category
        """
        recommendations = []

        # Helper to add recommendations
        def add_recommendation(text: str, category: str, priority: int):
            recommendations.append(
                {
                    "text": text,
                    "category": category,
                    "priority": priority,  # 1 = high, 2 = medium, 3 = low
                }
            )

        # Add meta tag recommendations
        for rec in meta_analysis.get("recommendations", []):
            add_recommendation(rec, "meta_tags", 1)

        # Add heading recommendations
        for rec in heading_analysis.get("recommendations", []):
            add_recommendation(rec, "headings", 2)

        # Add URL recommendations
        for rec in url_analysis.get("recommendations", []):
            add_recommendation(rec, "url_structure", 2)

        # Add keyword usage recommendations
        for rec in keyword_analysis.get("recommendations", []):
            add_recommendation(rec, "keyword_usage", 1)

        # Add optimization level recommendations
        for rec in optimization_analysis.get("recommendations", []):
            add_recommendation(rec, "optimization", 2)

        # Add content relevance recommendations
        for rec in content_relevance.get("recommendations", []):
            add_recommendation(rec, "content_relevance", 1)

        # Add LSI keyword recommendations
        for rec in lsi_analysis.get("recommendations", []):
            add_recommendation(rec, "lsi_keywords", 3)

        # Sort recommendations by priority
        recommendations.sort(key=lambda x: x["priority"])

        return recommendations

    def divide_text_into_sections(self, text: str, num_sections: int = 4) -> List[str]:
        """
        Divide text into equal sections for analyzing keyword distribution.

        Args:
            text: Text content
            num_sections: Number of sections to divide into

        Returns:
            List of text sections
        """
        words = text.split()
        section_size = max(1, len(words) // num_sections)
        sections = []

        for i in range(0, len(words), section_size):
            section = " ".join(words[i : i + section_size])
            sections.append(section)

        # Ensure we have exactly num_sections by combining small sections if needed
        while len(sections) > num_sections:
            sections[-2] = sections[-2] + " " + sections[-1]
            sections.pop()

        return sections

    def is_keyword_stuffed(self, text: str) -> bool:
        """
        Detect potential keyword stuffing in text.

        Args:
            text: Text to analyze

        Returns:
            Boolean indicating if text appears to be keyword stuffed
        """
        # Tokenize and count word frequencies
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words]

        if not words:
            return False

        word_counts = Counter(words)
        total_words = len(words)

        # Check if any word appears with high frequency
        for word, count in word_counts.most_common(5):
            frequency = count / total_words
            if (
                len(word) > 3 and frequency > 0.1
            ):  # If a non-trivial word appears more than 10% of the time
                return True

        # Check for suspicious repetition patterns
        sentences = text.split(".")
        if len(sentences) >= 3:
            # Look for repeated phrases across consecutive sentences
            for i in range(len(sentences) - 2):
                s1 = sentences[i].lower()
                s2 = sentences[i + 1].lower()
                s3 = sentences[i + 2].lower()

                # Extract 3-grams from each sentence
                trigrams1 = [" ".join(words[j : j + 3]) for j in range(len(words) - 2)]

                # Check if any 3-gram appears in all three consecutive sentences
                for trigram in trigrams1:
                    if trigram in s2 and trigram in s3 and len(trigram) > 10:
                        return True

        return False
