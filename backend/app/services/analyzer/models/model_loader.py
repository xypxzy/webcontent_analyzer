from typing import Dict, Any, Optional
import os
from loguru import logger
import spacy


class ModelLoader:
    """Utility class for loading and managing NLP models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model loader with configuration."""
        self.config = config or {}
        self.models = {}
        self.cache_dir = self.config.get("cache_dir")

        if self.cache_dir:
            os.environ["SPACY_MODEL_CACHE_DIR"] = self.cache_dir

        # Load spaCy models based on config
        self.spacy_models = {}
        self._load_spacy_models()

    def _load_spacy_models(self):
        """Load spaCy models specified in the configuration."""
        models_to_load = self.config.get(
            "spacy_models",
            {
                "en": "en_core_web_sm",
                "ru": "ru_core_news_sm",
            },
        )

        for lang, model_name in models_to_load.items():
            try:
                self.spacy_models[lang] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model {model_name} for language {lang}")
            except Exception as e:
                logger.error(f"Failed to load spaCy model {model_name}: {str(e)}")
                try:
                    # Fall back to blank model
                    self.spacy_models[lang] = spacy.blank(lang)
                    logger.info(f"Loaded blank spaCy model for {lang}")
                except Exception as inner_e:
                    logger.error(f"Failed to load any model for {lang}: {inner_e}")

    def get_spacy_model(self, lang: str):
        """Get a spaCy model for the specified language."""
        if lang in self.spacy_models:
            return self.spacy_models[lang]

        # If language not available, return English model as fallback
        if "en" in self.spacy_models:
            logger.warning(f"No model for {lang}, using English model instead")
            return self.spacy_models["en"]

        # If no models available, create a blank English model
        logger.warning(f"No models available, creating blank English model")
        return spacy.blank("en")

    def get_embedding_model(self, model_name: str = "default"):
        """Get a text embedding model."""
        # Placeholder for embedding models
        # In a real implementation, this would load models like BERT, Sentence Transformers, etc.
        return None

    def get_sentiment_model(self, lang: str = "en"):
        """Get a sentiment analysis model for the specified language."""
        # Placeholder for sentiment models
        return None
