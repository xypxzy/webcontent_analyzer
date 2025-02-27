from typing import List, Dict, Any, Set, Optional
import re
import string

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextProcessor:
    """Utility class for text processing tasks."""

    def __init__(self, language: str = "english"):
        """Initialize text processor."""
        self.language = language
        self.stopwords = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(string.punctuation)

    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of word tokens
        """
        return word_tokenize(text)

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.

        Args:
            text: Text to tokenize

        Returns:
            List of sentence tokens
        """
        return sent_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.

        Args:
            tokens: List of word tokens

        Returns:
            List of tokens with stopwords removed
        """
        return [word for word in tokens if word.lower() not in self.stopwords]

    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """
        Remove punctuation tokens.

        Args:
            tokens: List of word tokens

        Returns:
            List of non-punctuation tokens
        """
        return [word for word in tokens if word not in self.punctuation]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.

        Args:
            tokens: List of word tokens

        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def preprocess_text(
        self,
        text: str,
        remove_stops: bool = True,
        remove_punct: bool = True,
        lemmatize: bool = True,
    ) -> List[str]:
        """
        Fully preprocess text with configurable options.

        Args:
            text: Text to process
            remove_stops: Whether to remove stopwords
            remove_punct: Whether to remove punctuation
            lemmatize: Whether to lemmatize tokens

        Returns:
            List of processed tokens
        """
        # Tokenize
        tokens = self.tokenize_words(text)

        # Remove punctuation if requested
        if remove_punct:
            tokens = self.remove_punctuation(tokens)

        # Remove stopwords if requested
        if remove_stops:
            tokens = self.remove_stopwords(tokens)

        # Lemmatize if requested
        if lemmatize:
            tokens = self.lemmatize(tokens)

        return tokens

    def extract_keywords(self, text: str, n: int = 10) -> List[str]:
        """
        Extract potential keywords from text based on frequency and position.

        Args:
            text: Text to analyze
            n: Number of keywords to extract

        Returns:
            List of potential keywords
        """
        # Preprocess text
        tokens = self.preprocess_text(text)

        # Count frequencies
        word_freq = {}
        for token in tokens:
            if len(token) > 2:  # Ignore very short words
                word_freq[token] = word_freq.get(token, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # Extract top N keywords
        return [word for word, freq in sorted_words[:n]]

    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate basic readability metrics for text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of readability metrics
        """
        # Get sentences and words
        sentences = self.tokenize_sentences(text)
        words = self.tokenize_words(text)

        # Filter out punctuation-only tokens
        words = [w for w in words if any(c.isalpha() for c in w)]

        # Count syllables (basic approximation)
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            count = 0
            vowels = "aeiouy"
            if word[0] in vowels:
                count += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i - 1] not in vowels:
                    count += 1
            if word.endswith("e"):
                count -= 1
            if count == 0:
                count = 1
            return count

        syllables = sum(count_syllables(word) for word in words)

        # Calculate metrics
        num_sentences = len(sentences)
        num_words = len(words)

        if num_sentences == 0 or num_words == 0:
            return {
                "flesch_kincaid": 0.0,
                "syllables_per_word": 0.0,
                "words_per_sentence": 0.0,
            }

        words_per_sentence = num_words / num_sentences
        syllables_per_word = syllables / num_words

        # Flesch-Kincaid Grade Level
        flesch_kincaid = (
            (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
        )

        return {
            "flesch_kincaid": round(flesch_kincaid, 2),
            "syllables_per_word": round(syllables_per_word, 2),
            "words_per_sentence": round(words_per_sentence, 2),
        }

    def extract_ngrams(self, text: str, n: int) -> List[str]:
        """
        Extract n-grams from text.

        Args:
            text: Text to process
            n: Size of n-grams

        Returns:
            List of n-grams
        """
        words = self.tokenize_words(text.lower())
        words = [word for word in words if word.isalpha()]

        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.append(ngram)

        return ngrams
