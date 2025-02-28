from typing import Dict, Any, Optional
from loguru import logger
import nltk
from nltk.corpus import stopwords


class ModelLoader:
    """Utility class for loading and managing NLP models (NLTK based, no spaCy)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model loader with configuration."""
        self.config = config or {}
        self.models = {}
        self.cache_dir = self.config.get("cache_dir")

        # Load required NLTK resources
        self._load_nltk_resources()

        # Init language models
        self.lang_models = {}

    def _load_nltk_resources(self):
        """Load required NLTK resources."""
        try:
            # Download essential NLTK resources if they don't exist
            for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
                try:
                    nltk.data.find(f"tokenizers/{resource}")
                except LookupError:
                    nltk.download(resource)
                    logger.info(f"Downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.error(f"Failed to download NLTK resources: {str(e)}")

    def get_spacy_model(self, lang: str):
        """
        Get a language processing model for the specified language.
        This is a compatibility wrapper for code that expected spaCy models.

        Returns a simple wrapper object with basic tokenization capabilities.
        """
        if lang not in self.lang_models:
            self.lang_models[lang] = SimpleNLPModel(lang)
            logger.info(f"Created simple NLP model for {lang}")

        return self.lang_models[lang]

    def get_embedding_model(self, model_name: str = "default"):
        """Get a text embedding model."""
        # Simple TF-IDF based embedding as placeholder
        return None

    def get_sentiment_model(self, lang: str = "en"):
        """Get a sentiment analysis model for the specified language."""
        # Returns None as placeholder
        return None


class SimpleNLPModel:
    """A simple class to mimic some basic spaCy functionality."""

    def __init__(self, lang: str):
        """Initialize with language code."""
        self.lang = lang
        # Load stopwords for this language if available
        try:
            self.stop_words = set(stopwords.words(self._get_full_lang_name(lang)))
        except:
            self.stop_words = set()
            logger.warning(f"Could not load stopwords for {lang}")

    def _get_full_lang_name(self, lang_code: str) -> str:
        """Convert language code to full name for NLTK."""
        lang_map = {
            "en": "english",
            "ru": "russian",
            "fr": "french",
            "de": "german",
            "es": "spanish",
            "it": "italian",
            "nl": "dutch",
            "pt": "portuguese"
        }
        return lang_map.get(lang_code, "english")

    def __call__(self, text: str):
        """Process text and return a simple document-like object."""
        return SimpleDoc(text, self.lang, self.stop_words)


class SimpleDoc:
    """A simple class to mimic basic spaCy Doc functionality."""

    def __init__(self, text: str, lang: str, stop_words: set):
        """Initialize with text and language."""
        self.text = text
        self.lang = lang
        self.stop_words = stop_words

        # Tokenize the text
        self.tokens = nltk.word_tokenize(text)

        # Create simple token objects
        self.doc_tokens = [SimpleToken(t, i, stop_words) for i, t in enumerate(self.tokens)]

        # Extract sentences
        self.sents = [SimpleSentence(s) for s in nltk.sent_tokenize(text)]

        # Basic NER (just capitalized words)
        self.ents = self._extract_basic_entities()

    def _extract_basic_entities(self):
        """Extract simple entities based on capitalization."""
        entities = []
        for i, token in enumerate(self.tokens):
            if token[0].isupper() and len(token) > 1 and token.lower() not in self.stop_words:
                start_char = self.text.find(token)
                if start_char >= 0:  # If token is found in text
                    entities.append(SimpleEntity(
                        text=token,
                        start_char=start_char,
                        end_char=start_char + len(token),
                        label="ENTITY"  # Simple approach
                    ))
        return entities

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.doc_tokens[i]

    def similarity(self, other):
        """Calculate simple text similarity."""
        # Implement a very basic similarity metric
        # In a real implementation, you might use something more sophisticated
        words1 = set(t.lower() for t in self.tokens if t.isalpha())

        if isinstance(other, SimpleDoc):
            words2 = set(t.lower() for t in other.tokens if t.isalpha())
        else:
            # Assume it's a string
            words2 = set(t.lower() for t in nltk.word_tokenize(other) if t.isalpha())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


class SimpleToken:
    """A simple class to mimic basic spaCy Token functionality."""

    def __init__(self, text: str, i: int, stop_words: set):
        """Initialize with token text and position."""
        self.text = text
        self.i = i
        self.lemma_ = text.lower()  # Simple lemmatization
        self.pos_ = "X"  # Default POS tag
        self.is_stop = text.lower() in stop_words
        self.is_alpha = text.isalpha()

    def __str__(self):
        return self.text


class SimpleSentence:
    """A simple class to mimic basic spaCy Span for sentences."""

    def __init__(self, text: str):
        """Initialize with sentence text."""
        self.text = text
        self.start_char = 0  # In a real implementation, this would be the actual offset
        self.end_char = len(text)


class SimpleEntity:
    """A simple class to mimic basic spaCy Entity."""

    def __init__(self, text: str, start_char: int, end_char: int, label: str):
        """Initialize with entity information."""
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.label_ = label