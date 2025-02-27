import asyncio
import re
from typing import Dict, List, Set, Tuple, Any, Optional
import os
from functools import lru_cache

import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
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


class TextPreprocessor:
    """
    Класс для предварительной обработки текста перед анализом.
    Выполняет нормализацию, токенизацию, лемматизацию и другие операции.
    """

    # Поддерживаемые языки и соответствующие модели spaCy
    SUPPORTED_LANGUAGES = {
        "en": "en_core_web_sm",
        "ru": "ru_core_news_sm",
        # Можно добавить другие языки при необходимости
    }

    def __init__(self, lang: str = "en", cache_dir: Optional[str] = None):
        """
        Инициализация предпроцессора.

        Args:
            lang: Код языка по умолчанию ('en', 'ru', и т.д.)
            cache_dir: Директория для кэширования моделей
        """
        self.default_lang = lang
        self.cache_dir = cache_dir

        # Загружаем стоп-слова для основных языков
        self.stopwords = {}
        self._load_stopwords()

        # Загружаем модели spaCy для поддерживаемых языков
        self.nlp_models = {}
        self._load_spacy_models()

    def _load_stopwords(self):
        """Загрузка стоп-слов для разных языков из NLTK"""
        from nltk.corpus import stopwords

        try:
            for lang in ["english", "russian"]:
                lang_code = "en" if lang == "english" else "ru"
                self.stopwords[lang_code] = set(stopwords.words(lang))
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

    def _load_spacy_models(self):
        """Загрузка моделей spaCy для поддерживаемых языков"""
        # Установка переменной окружения для кэша моделей
        if self.cache_dir:
            os.environ["SPACY_MODEL_CACHE_DIR"] = self.cache_dir

        for lang_code, model_name in self.SUPPORTED_LANGUAGES.items():
            try:
                # Проверяем, установлена ли модель
                try:
                    self.nlp_models[lang_code] = spacy.load(model_name)
                    logger.info(
                        f"Загружена модель spaCy {model_name} для языка {lang_code}"
                    )
                except OSError:
                    logger.warning(
                        f"Модель {model_name} не найдена, пытаемся загрузить..."
                    )
                    # Если модель не установлена, пытаемся загрузить базовую
                    self.nlp_models[lang_code] = spacy.blank(lang_code)
                    logger.info(f"Загружена базовая модель spaCy для языка {lang_code}")
            except Exception as e:
                logger.error(f"Не удалось загрузить модель для {lang_code}: {str(e)}")

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

        # Токенизация, лемматизация, POS-теггинг с помощью spaCy
        nlp_result = await self._process_with_spacy(normalized_text, lang)

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
            "doc": nlp_result["doc"],  # spaCy Doc для дальнейшего использования
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

    async def _process_with_spacy(self, text: str, lang: str) -> Dict[str, Any]:
        """
        Обработка текста с помощью spaCy.

        Args:
            text: Нормализованный текст
            lang: Код языка

        Returns:
            Словарь с результатами обработки
        """
        # Если нет модели для указанного языка, используем английскую
        if lang not in self.nlp_models:
            lang = "en"
            logger.warning(
                f"Модель для языка {lang} не найдена, используем английскую модель"
            )

        # Обработка в отдельном потоке, так как spaCy не асинхронный
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(None, self.nlp_models[lang], text)

        # Извлечение токенов, лемм, POS-тегов и именованных сущностей
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        pos_tags = [token.pos_ for token in doc]

        # Фильтрация стоп-слов
        stopwords = self.stopwords.get(lang, set())
        tokens_without_stopwords = [
            token.text for token in doc if token.text.lower() not in stopwords
        ]
        lemmas_without_stopwords = [
            token.lemma_ for token in doc if token.text.lower() not in stopwords
        ]

        # Извлечение именованных сущностей
        entities = [
            {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
            }
            for ent in doc.ents
        ]

        return {
            "tokens": tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "tokens_without_stopwords": tokens_without_stopwords,
            "lemmas_without_stopwords": lemmas_without_stopwords,
            "entities": entities,
            "doc": doc,  # Сохраняем объект Doc для дальнейшего использования
        }

    def add_custom_stopwords(self, stopwords: List[str], lang: str = "en"):
        """
        Добавление пользовательских стоп-слов.

        Args:
            stopwords: Список стоп-слов для добавления
            lang: Код языка
        """
        if lang not in self.stopwords:
            self.stopwords[lang] = set()

        self.stopwords[lang].update(stopwords)
        logger.info(
            f"Добавлено {len(stopwords)} пользовательских стоп-слов для языка {lang}"
        )
