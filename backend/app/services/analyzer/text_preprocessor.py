import asyncio
import re
from typing import Dict, List, Set, Tuple, Any, Optional
import os
from functools import lru_cache

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from langdetect import detect, LangDetectException
from loguru import logger

# Загрузка необходимых ресурсов NLTK
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")


class TextPreprocessor:
    """
    Класс для предварительной обработки текста перед анализом.
    Выполняет нормализацию, токенизацию, лемматизацию и другие операции.
    NLTK-based implementation (no spaCy dependency).
    """

    # Поддерживаемые языки
    SUPPORTED_LANGUAGES = {
        "en": "english",
        "ru": "russian",
        # Можно добавить другие языки при необходимости
    }

    def __init__(self, lang: str = "en", cache_dir: Optional[str] = None):
        """
        Инициализация предпроцессора.

        Args:
            lang: Код языка по умолчанию ('en', 'ru', и т.д.)
            cache_dir: Директория для кэширования моделей (unused in NLTK version)
        """
        self.default_lang = lang
        self.cache_dir = cache_dir

        # Загружаем стоп-слова для основных языков
        self.stopwords = {}
        self._load_stopwords()

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

    def _load_stopwords(self):
        """Загрузка стоп-слов для разных языков из NLTK"""
        from nltk.corpus import stopwords

        try:
            for lang_code, lang_name in self.SUPPORTED_LANGUAGES.items():
                try:
                    self.stopwords[lang_code] = set(stopwords.words(lang_name))
                except:
                    # If language not available, use a minimal set
                    if lang_code == "en":
                        self.stopwords[lang_code] = set(
                            ["a", "an", "the", "in", "on", "at", "of", "and", "or", "to", "for"]
                        )
                    elif lang_code == "ru":
                        self.stopwords[lang_code] = set(
                            ["и", "в", "на", "с", "по", "у", "к", "о", "из", "не", "что"]
                        )

            logger.info(f"Загружены стоп-слова для {len(self.stopwords)} языков")
        except Exception as e:
            logger.error(f"Ошибка при загрузке стоп-слов: {str(e)}")
            # Если не удалось загрузить из NLTK, используем минимальный набор
            self.stopwords = {
                "en": set(
                    ["a", "an", "the", "in", "on", "at", "of", "and", "or", "to", "for"]
                ),
                "ru": set(
                    ["и", "в", "на", "с", "по", "у", "к", "о", "из", "не", "что"]
                ),
            }

    def detect_language(self, text: str) -> str:
        """
        Определение языка текста.

        Args:
            text: Текст для определения языка

        Returns:
            Код языка (en, ru и т.д.) или "unknown" если не удалось определить
        """
        if not text or len(text.strip()) < 10:
            return self.default_lang

        try:
            # Используем только первые 1000 символов для ускорения
            sample_text = text[:1000].strip()
            detected = detect(sample_text)
            # Преобразуем 2-символьный код языка, если необходимо
            if detected == "en":
                return "en"
            elif detected in ["ru", "uk", "be"]:
                return "ru"  # Для близких языков используем общую модель
            else:
                # Если язык не поддерживается, используем английский
                if detected not in self.SUPPORTED_LANGUAGES:
                    logger.warning(
                        f"Обнаружен неподдерживаемый язык: {detected}, используем {self.default_lang}"
                    )
                    return self.default_lang
                return detected
        except LangDetectException:
            logger.warning(
                "Не удалось определить язык текста, используем язык по умолчанию"
            )
            return self.default_lang

    async def preprocess(self, text: str, lang: Optional[str] = None) -> Dict[str, Any]:
        """
        Асинхронная предобработка текста.

        Args:
            text: Исходный текст для обработки
            lang: Код языка (если None, будет определен автоматически)

        Returns:
            Словарь с обработанным текстом и результатами предобработки
        """
        # Определение языка, если не указан
        if lang is None:
            lang = self.detect_language(text)

        # Нормализация текста (удаление лишних пробелов, приведение к нижнему регистру)
        normalized_text = self._normalize_text(text)

        # Токенизация на уровне предложений
        sentences = sent_tokenize(normalized_text)

        # Используем NLTK для токенизации, лемматизации и т.д.
        nlp_result = await self._process_with_nltk(normalized_text, lang)

        # Подготовка результатов
        result = {
            "lang": lang,
            "original_text": text,
            "normalized_text": normalized_text,
            "sentences": sentences,
            "tokens": nlp_result["tokens"],
            "lemmas": nlp_result["lemmas"],
            "pos_tags": nlp_result["pos_tags"],
            "tokens_without_stopwords": nlp_result["tokens_without_stopwords"],
            "lemmas_without_stopwords": nlp_result["lemmas_without_stopwords"],
            "entities": nlp_result["entities"],
            "doc": None,  # В NLTK версии нет прямого аналога spaCy Doc
        }

        return result

    def _normalize_text(self, text: str) -> str:
        """
        Нормализация текста.

        Args:
            text: Исходный текст

        Returns:
            Нормализованный текст
        """
        # Удаление множественных пробелов и переносов строк
        text = re.sub(r"\s+", " ", text)
        # Удаление специальных символов и HTML-тегов (если остались)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^\w\s.,!?;:\'\"()\[\]\{\}«»–—-]", " ", text)
        # Замена множественных пробелов одиночными
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def _process_with_nltk(self, text: str, lang: str) -> Dict[str, Any]:
        """
        Обработка текста с помощью NLTK.

        Args:
            text: Нормализованный текст
            lang: Код языка

        Returns:
            Словарь с результатами обработки
        """
        # Если язык не поддерживается, используем английский
        if lang not in self.SUPPORTED_LANGUAGES:
            lang = "en"
            logger.warning(
                f"Язык {lang} не поддерживается, используем английский"
            )

        # Выполняем токенизацию и POS-теггинг в отдельном потоке
        loop = asyncio.get_event_loop()

        tokens_result = await loop.run_in_executor(None, word_tokenize, text)

        # POS tagging (supports English best)
        if lang == "en":
            pos_tags_result = await loop.run_in_executor(None, pos_tag, tokens_result)
            pos_tags = [tag for _, tag in pos_tags_result]
        else:
            # For non-English, just put a placeholder tag
            pos_tags = ["X"] * len(tokens_result)

        # Lemmatization
        lemmas = []
        for token in tokens_result:
            lemma = await loop.run_in_executor(None, self.lemmatizer.lemmatize, token)
            lemmas.append(lemma)

        # Фильтрация стоп-слов
        stopwords_set = self.stopwords.get(lang, set())
        tokens_without_stopwords = [
            token for token in tokens_result if token.lower() not in stopwords_set
        ]
        lemmas_without_stopwords = [
            token for token, original in zip(lemmas, tokens_result)
            if original.lower() not in stopwords_set
        ]

        # Simplified NER (Named Entity Recognition)
        # This is very basic - only uses capitalization as a heuristic
        entities = []
        for i, token in enumerate(tokens_result):
            if token[0].isupper() and len(token) > 1 and token.lower() not in stopwords_set:
                entity_type = "ENTITY"  # Simple approach, no detailed classification
                entities.append({
                    "text": token,
                    "start": text.find(token),
                    "end": text.find(token) + len(token),
                    "label": entity_type,
                })

        return {
            "tokens": tokens_result,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "tokens_without_stopwords": tokens_without_stopwords,
            "lemmas_without_stopwords": lemmas_without_stopwords,
            "entities": entities,
            "doc": None,  # No equivalent in NLTK
        }

    def add_custom_stopwords(self, stopwords_list: List[str], lang: str = "en"):
        """
        Добавление пользовательских стоп-слов.

        Args:
            stopwords_list: Список стоп-слов для добавления
            lang: Код языка
        """
        if lang not in self.stopwords:
            self.stopwords[lang] = set()

        self.stopwords[lang].update(stopwords_list)
        logger.info(
            f"Добавлено {len(stopwords_list)} пользовательских стоп-слов для языка {lang}"
        )