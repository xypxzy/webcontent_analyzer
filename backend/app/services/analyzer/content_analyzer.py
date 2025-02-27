import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Set
import os
import json

from loguru import logger
import spacy
from bs4 import BeautifulSoup

from app.services.analyzer.text_preprocessor import TextPreprocessor
from app.services.analyzer.basic_analyzer import BasicTextAnalyzer
from app.services.analyzer.semantic_analyzer import SemanticAnalyzer
from app.services.analyzer.sentiment_analyzer import SentimentAnalyzer
from app.services.analyzer.seo_analyzer import SEOAnalyzer
from app.services.analyzer.models.model_loader import ModelLoader
from app.core.config import settings


class ContentAnalyzer:
    """
    Главный класс для анализа текстового контента веб-страниц.
    Координирует работу всех специализированных анализаторов.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация анализатора контента.

        Args:
            config: Опциональная конфигурация для настройки анализаторов
        """
        self.config = config or {}
        self.models = ModelLoader(self.config.get("models_config", {}))

        # Настройка параметров анализа
        self.default_language = self.config.get(
            "language", settings.NLP_DEFAULT_LANGUAGE
        )
        self.max_text_length = self.config.get("max_text_length", 100000)
        self.enable_cache = self.config.get("enable_cache", True)

        # Инициализация предварительного процессора и анализаторов
        self.preprocessor = TextPreprocessor(
            lang=self.default_language, cache_dir=settings.NLP_MODELS_CACHE_DIR
        )

        # Инициализация специализированных анализаторов
        self.basic_analyzer = BasicTextAnalyzer(
            lang=self.default_language, models=self.models
        )

        self.semantic_analyzer = SemanticAnalyzer(
            lang=self.default_language,
            models=self.models,
            config=self.config.get("semantic_config", {}),
        )

        self.sentiment_analyzer = SentimentAnalyzer(
            lang=self.default_language,
            models=self.models,
            config=self.config.get("sentiment_config", {}),
        )

        self.seo_analyzer = SEOAnalyzer(
            lang=self.default_language, config=self.config.get("seo_config", {})
        )

        # Внутреннее хранилище для кэширования результатов
        self._cache = {}

    async def analyze_content(
        self,
        text: str,
        html: str = None,
        metadata: Dict[str, Any] = None,
        url: str = None,
        structure: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Выполнение полного анализа текстового контента.

        Args:
            text: Основной текст для анализа
            html: HTML-контент страницы (опционально)
            metadata: Метаданные страницы (опционально)
            url: URL страницы (опционально)
            structure: Структура страницы (опционально)

        Returns:
            Словарь с результатами всех типов анализа
        """
        # Проверка кэша если включен
        cache_key = self._get_cache_key(text, url)
        if self.enable_cache and cache_key in self._cache:
            logger.info(
                f"Используем кэшированные результаты анализа для {url or 'текста'}"
            )
            return self._cache[cache_key]

        # Ограничение длины текста для производительности
        if len(text) > self.max_text_length:
            logger.warning(
                f"Текст слишком длинный ({len(text)} символов), обрезаем до {self.max_text_length}"
            )
            text = text[: self.max_text_length]

        # Определение языка текста, если не указан явно
        detected_lang = self.preprocessor.detect_language(text)
        lang = self.default_language if detected_lang == "unknown" else detected_lang

        # Предварительная обработка текста
        logger.info(f"Начало обработки текста ({len(text)} символов)")
        start_time = datetime.now()

        processed_text = await self.preprocessor.preprocess(text=text, lang=lang)

        # Выполнение всех типов анализа параллельно
        tasks = [
            self._run_basic_analysis(processed_text, lang),
            self._run_semantic_analysis(processed_text, text, lang),
            self._run_sentiment_analysis(processed_text, text, lang),
            self._run_seo_analysis(processed_text, html, metadata, url, structure),
        ]

        # Выполнение всех задач параллельно
        results = await asyncio.gather(*tasks)

        # Объединение результатов
        analysis_result = {
            "basic_metrics": results[0],
            "semantic_analysis": results[1],
            "sentiment_analysis": results[2],
            "seo_metrics": results[3],
            "language": lang,
            "analysis_time": (datetime.now() - start_time).total_seconds(),
            "recommendations": [],
        }

        # Генерация рекомендаций на основе всех результатов анализа
        analysis_result["recommendations"] = await self._generate_recommendations(
            analysis_result
        )

        # Сохраняем результаты в кэш, если кэширование включено
        if self.enable_cache:
            self._cache[cache_key] = analysis_result

        logger.info(
            f"Анализ контента завершен за {analysis_result['analysis_time']:.2f} секунд"
        )
        return analysis_result

    async def _run_basic_analysis(self, processed_text, lang):
        """Запуск базового анализа текста"""
        logger.info("Выполнение базового анализа текста")
        return await self.basic_analyzer.analyze(processed_text, lang)

    async def _run_semantic_analysis(self, processed_text, raw_text, lang):
        """Запуск семантического анализа"""
        logger.info("Выполнение семантического анализа")
        return await self.semantic_analyzer.analyze(processed_text, raw_text, lang)

    async def _run_sentiment_analysis(self, processed_text, raw_text, lang):
        """Запуск анализа тональности и UX-метрик"""
        logger.info("Выполнение анализа тональности")
        return await self.sentiment_analyzer.analyze(processed_text, raw_text, lang)

    async def _run_seo_analysis(self, processed_text, html, metadata, url, structure):
        """Запуск SEO-анализа"""
        logger.info("Выполнение SEO-анализа")
        return await self.seo_analyzer.analyze(
            processed_text=processed_text,
            html_content=html,
            metadata=metadata or {},
            url=url,
            structure=structure or {},
        )

    async def _generate_recommendations(
        self, analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Генерация рекомендаций на основе результатов анализа.

        Args:
            analysis_result: Результаты всех типов анализа

        Returns:
            Список рекомендаций по улучшению контента
        """
        recommendations = []

        # Получение рекомендаций от всех анализаторов
        basic_recs = await self.basic_analyzer.generate_recommendations(
            analysis_result["basic_metrics"]
        )
        semantic_recs = await self.semantic_analyzer.generate_recommendations(
            analysis_result["semantic_analysis"]
        )
        sentiment_recs = await self.sentiment_analyzer.generate_recommendations(
            analysis_result["sentiment_analysis"]
        )
        seo_recs = await self.seo_analyzer.generate_recommendations(
            analysis_result["seo_metrics"]
        )

        # Объединение всех рекомендаций
        recommendations.extend(basic_recs)
        recommendations.extend(semantic_recs)
        recommendations.extend(sentiment_recs)
        recommendations.extend(seo_recs)

        # Сортировка рекомендаций по приоритету
        recommendations.sort(key=lambda x: x.get("priority", 3))

        return recommendations

    def _get_cache_key(self, text: str, url: Optional[str] = None) -> str:
        """Создание ключа для кэширования результатов"""
        import hashlib

        if url:
            return f"url_{hashlib.md5(url.encode()).hexdigest()}"
        else:
            # Для текста берем только первые 1000 символов для хэша
            return f"text_{hashlib.md5(text[:1000].encode()).hexdigest()}"

    def clear_cache(self):
        """Очистка кэша результатов анализа"""
        self._cache.clear()
