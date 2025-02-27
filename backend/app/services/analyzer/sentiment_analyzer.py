import asyncio
from typing import Dict, List, Any
import re
from collections import Counter

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import textblob
from loguru import logger
import numpy as np

from app.services.analyzer.models.model_loader import ModelLoader


class SentimentAnalyzer:
    """
    Класс для анализа тональности текста и UX-метрик.
    Выполняет определение общей тональности, эмоциональный анализ,
    анализ CTA-элементов и другие UX-метрики.
    """

    # Словари для анализа эмоций
    EMOTION_KEYWORDS = {
        "joy": [
            "счастье",
            "радость",
            "восторг",
            "ликование",
            "веселье",
            "улыбка",
            "смех",
            "наслаждение",
            "happy",
            "joy",
            "delighted",
            "excited",
            "pleasure",
            "smile",
            "laugh",
            "enjoy",
        ],
        "sadness": [
            "грусть",
            "печаль",
            "тоска",
            "уныние",
            "скорбь",
            "слезы",
            "горе",
            "сожаление",
            "sad",
            "sorrow",
            "grief",
            "depressed",
            "unhappy",
            "tears",
            "regret",
            "upset",
        ],
        "anger": [
            "гнев",
            "злость",
            "ярость",
            "раздражение",
            "возмущение",
            "ненависть",
            "бешенство",
            "angry",
            "furious",
            "rage",
            "irritated",
            "annoyed",
            "hate",
            "mad",
            "outraged",
        ],
        "fear": [
            "страх",
            "ужас",
            "тревога",
            "опасение",
            "паника",
            "испуг",
            "боязнь",
            "фобия",
            "fear",
            "terror",
            "anxiety",
            "worry",
            "panic",
            "scared",
            "frightened",
            "dread",
        ],
        "surprise": [
            "удивление",
            "изумление",
            "шок",
            "неожиданность",
            "поразительно",
            "surprise",
            "astonishment",
            "amazement",
            "shock",
            "unexpected",
            "wow",
            "astonished",
        ],
    }

    # Словари для анализа CTA
    CTA_PATTERNS = [
        r"купи[а-я]*\s+сейчас",
        r"заказ[а-я]*\s+сейчас",
        r"оформи[а-я]*\s+заказ",
        r"получи[а-я]*\s+сейчас",
        r"регистрир[а-я]*\s+сейчас",
        r"подпис[а-я]*\s+сейчас",
        r"начни[а-я]*\s+сейчас",
        r"попроб[а-я]*\s+бесплатно",
        r"скач[а-я]*\s+сейчас",
        r"забронир[а-я]*\s+сейчас",
        r"записа[а-я]*\s+сейчас",
        r"присоедин[а-я]*\s+сейчас",
        r"узнай[а-я]*\s+больше",
        r"buy\s+now",
        r"order\s+now",
        r"get\s+started",
        r"sign\s+up",
        r"register\s+now",
        r"download\s+now",
        r"subscribe\s+now",
        r"try\s+for\s+free",
        r"book\s+now",
        r"join\s+now",
        r"learn\s+more",
    ]

    # Психологические триггеры в CTA
    CTA_TRIGGERS = {
        "urgency": [
            "сейчас",
            "немедленно",
            "сегодня",
            "до конца",
            "успей",
            "ограниченно",
            "срочно",
            "now",
            "today",
            "immediately",
            "limited",
            "hurry",
            "urgent",
            "deadline",
        ],
        "scarcity": [
            "ограниченно",
            "только",
            "осталось",
            "эксклюзивно",
            "редкий",
            "последний",
            "limited",
            "only",
            "exclusive",
            "rare",
            "last chance",
            "few left",
            "exclusive",
        ],
        "curiosity": [
            "узнай",
            "открой",
            "раскрой",
            "секрет",
            "интересно",
            "любопытно",
            "discover",
            "secret",
            "revealed",
            "find out",
            "interesting",
            "curiosity",
        ],
        "value": [
            "бесплатно",
            "скидка",
            "выгодно",
            "экономия",
            "прибыль",
            "преимущество",
            "free",
            "discount",
            "save",
            "profit",
            "benefit",
            "advantage",
            "value",
        ],
        "social_proof": [
            "популярный",
            "рекомендуют",
            "проверенный",
            "одобренный",
            "доверяют",
            "popular",
            "recommended",
            "trusted",
            "approved",
            "bestseller",
            "trending",
        ],
        "fear_of_missing_out": [
            "не упусти",
            "не пропусти",
            "шанс",
            "однажды",
            "уникальный",
            "don't miss",
            "chance",
            "opportunity",
            "once",
            "unique",
            "limited time",
        ],
    }

    def __init__(self, lang: str = "en", models=None, config: Dict[str, Any] = None):
        """
        Инициализация анализатора тональности.

        Args:
            lang: Код языка по умолчанию ('en', 'ru', и т.д.)
            models: Загрузчик моделей
            config: Конфигурация анализатора
        """
        self.default_lang = lang
        self.models = models
        self.config = config or {}

        # Компиляция регулярных выражений для CTA
        self.cta_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.CTA_PATTERNS
        ]

        # Инициализация анализатора тональности VADER для английского языка
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")

        self.sia = SentimentIntensityAnalyzer()

        # Параметры конфигурации
        self.min_sentence_length = self.config.get("min_sentence_length", 5)
        self.aspect_threshold = self.config.get("aspect_threshold", 0.3)

    async def analyze(
        self, processed_text: Dict[str, Any], raw_text: str, lang: str
    ) -> Dict[str, Any]:
        """
        Выполнение анализа тональности и UX-метрик.

        Args:
            processed_text: Предварительно обработанный текст
            raw_text: Исходный необработанный текст
            lang: Код языка

        Returns:
            Словарь с результатами анализа тональности и UX-метрик
        """
        # Запускаем различные типы анализа параллельно
        overall_sentiment_task = asyncio.create_task(
            self._analyze_overall_sentiment(processed_text, lang)
        )

        sentence_sentiment_task = asyncio.create_task(
            self._analyze_sentence_sentiment(processed_text, lang)
        )

        emotion_analysis_task = asyncio.create_task(
            self._analyze_emotions(processed_text)
        )

        cta_analysis_task = asyncio.create_task(self._analyze_cta(raw_text))

        ux_metrics_task = asyncio.create_task(
            self._analyze_ux_metrics(processed_text, raw_text)
        )

        # Ожидаем завершения всех задач
        overall_sentiment = await overall_sentiment_task
        sentence_sentiment = await sentence_sentiment_task
        emotion_analysis = await emotion_analysis_task
        cta_analysis = await cta_analysis_task
        ux_metrics = await ux_metrics_task

        # Объединяем результаты
        result = {
            "overall_sentiment": overall_sentiment,
            "sentence_sentiment": sentence_sentiment,
            "emotion_analysis": emotion_analysis,
            "cta_analysis": cta_analysis,
            "ux_metrics": ux_metrics,
        }

        return result

    async def _analyze_overall_sentiment(
        self, processed_text: Dict[str, Any], lang: str
    ) -> Dict[str, Any]:
        """
        Анализ общей тональности текста.

        Args:
            processed_text: Предварительно обработанный текст
            lang: Код языка

        Returns:
            Словарь с общей тональностью текста
        """
        text = processed_text.get("normalized_text", "")

        # Для английского используем VADER
        if lang == "en":
            try:
                # VADER анализ выполняется в отдельном потоке
                loop = asyncio.get_event_loop()
                sentiment_scores = await loop.run_in_executor(
                    None, self.sia.polarity_scores, text
                )

                # Определение тональности на основе compound score
                if sentiment_scores["compound"] >= 0.05:
                    sentiment = "positive"
                elif sentiment_scores["compound"] <= -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

                return {
                    "sentiment": sentiment,
                    "scores": {
                        "positive": sentiment_scores["pos"],
                        "negative": sentiment_scores["neg"],
                        "neutral": sentiment_scores["neu"],
                        "compound": sentiment_scores["compound"],
                    },
                }
            except Exception as e:
                logger.error(f"Ошибка при анализе тональности VADER: {str(e)}")

        # Для всех языков используем TextBlob (хотя он лучше работает с английским)
        try:
            loop = asyncio.get_event_loop()
            blob = await loop.run_in_executor(None, textblob.TextBlob, text)

            # TextBlob возвращает полярность в диапазоне [-1, 1]
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Определение тональности
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "scores": {
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "positive": max(0, polarity),
                    "negative": max(0, -polarity),
                    "neutral": 1 - abs(polarity),
                },
            }
        except Exception as e:
            logger.error(f"Ошибка при анализе тональности TextBlob: {str(e)}")

            # Возвращаем нейтральную тональность в случае ошибки
            return {
                "sentiment": "neutral",
                "scores": {
                    "positive": 0.33,
                    "negative": 0.33,
                    "neutral": 0.34,
                    "error": str(e),
                },
            }

    async def _analyze_sentence_sentiment(
        self, processed_text: Dict[str, Any], lang: str
    ) -> Dict[str, Any]:
        """
        Анализ тональности на уровне предложений.

        Args:
            processed_text: Предварительно обработанный текст
            lang: Код языка

        Returns:
            Словарь с тональностью по предложениям
        """
        sentences = processed_text.get("sentences", [])

        # Если текст слишком короткий, не анализируем по предложениям
        if not sentences:
            return {
                "sentences": [],
                "distribution": {"positive": 0, "negative": 0, "neutral": 1},
            }

        # Анализ тональности для каждого предложения
        sentence_results = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for i, sentence in enumerate(sentences):
            # Пропускаем слишком короткие предложения
            if len(sentence.split()) < self.min_sentence_length:
                continue

            # Для английского используем VADER
            if lang == "en":
                try:
                    loop = asyncio.get_event_loop()
                    sentiment_scores = await loop.run_in_executor(
                        None, self.sia.polarity_scores, sentence
                    )

                    # Определение тональности на основе compound score
                    if sentiment_scores["compound"] >= 0.05:
                        sentiment = "positive"
                        positive_count += 1
                    elif sentiment_scores["compound"] <= -0.05:
                        sentiment = "negative"
                        negative_count += 1
                    else:
                        sentiment = "neutral"
                        neutral_count += 1

                    sentence_results.append(
                        {
                            "index": i,
                            "text": sentence,
                            "sentiment": sentiment,
                            "scores": {
                                "positive": sentiment_scores["pos"],
                                "negative": sentiment_scores["neg"],
                                "neutral": sentiment_scores["neu"],
                                "compound": sentiment_scores["compound"],
                            },
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Ошибка при анализе тональности предложения VADER: {str(e)}"
                    )
            else:
                # Для всех языков используем TextBlob
                try:
                    loop = asyncio.get_event_loop()
                    blob = await loop.run_in_executor(None, textblob.TextBlob, sentence)

                    # TextBlob возвращает полярность в диапазоне [-1, 1]
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity

                    # Определение тональности
                    if polarity > 0.1:
                        sentiment = "positive"
                        positive_count += 1
                    elif polarity < -0.1:
                        sentiment = "negative"
                        negative_count += 1
                    else:
                        sentiment = "neutral"
                        neutral_count += 1

                    sentence_results.append(
                        {
                            "index": i,
                            "text": sentence,
                            "sentiment": sentiment,
                            "scores": {
                                "polarity": polarity,
                                "subjectivity": subjectivity,
                            },
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Ошибка при анализе тональности предложения TextBlob: {str(e)}"
                    )

        # Расчет распределения тональности
        total_sentences = positive_count + negative_count + neutral_count
        distribution = {
            "positive": positive_count / total_sentences if total_sentences > 0 else 0,
            "negative": negative_count / total_sentences if total_sentences > 0 else 0,
            "neutral": neutral_count / total_sentences if total_sentences > 0 else 1,
        }

        return {"sentences": sentence_results, "distribution": distribution}

    async def _analyze_emotions(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализ эмоциональной окраски текста.

        Args:
            processed_text: Предварительно обработанный текст

        Returns:
            Словарь с результатами эмоционального анализа
        """
        tokens = processed_text.get("tokens", [])
        lemmas = processed_text.get("lemmas", [])
        sentences = processed_text.get("sentences", [])

        # Счетчики эмоций
        emotion_counts = {emotion: 0 for emotion in self.EMOTION_KEYWORDS.keys()}
        emotion_sentences = {emotion: [] for emotion in self.EMOTION_KEYWORDS.keys()}

        # Проходим по каждому слову и ищем эмоциональные маркеры
        for lemma in lemmas:
            lemma_lower = lemma.lower()
            for emotion, keywords in self.EMOTION_KEYWORDS.items():
                if lemma_lower in keywords:
                    emotion_counts[emotion] += 1

        # Проверяем предложения на наличие эмоциональных маркеров
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            for emotion, keywords in self.EMOTION_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in sentence_lower:
                        emotion_sentences[emotion].append(
                            {"index": i, "text": sentence}
                        )
                        break  # Достаточно одного совпадения для предложения

        # Нормализация счетчиков эмоций
        total_emotion_words = sum(emotion_counts.values())

        # Расчет эмоционального профиля
        emotion_profile = {}
        for emotion, count in emotion_counts.items():
            emotion_profile[emotion] = (
                count / total_emotion_words if total_emotion_words > 0 else 0
            )

        # Определение доминирующей эмоции
        dominant_emotion = (
            max(emotion_profile.items(), key=lambda x: x[1])[0]
            if emotion_profile
            else None
        )

        # Интенсивность эмоций
        emotion_intensity = total_emotion_words / len(tokens) if tokens else 0

        return {
            "emotion_profile": emotion_profile,
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": emotion_intensity,
            "emotion_sentences": emotion_sentences,
        }

    async def _analyze_cta(self, text: str) -> Dict[str, Any]:
        """
        Анализ CTA-элементов (призывов к действию).

        Args:
            text: Текст для анализа

        Returns:
            Словарь с результатами анализа CTA
        """
        text_lower = text.lower()

        # Поиск CTA по шаблонам
        cta_matches = []
        for i, pattern in enumerate(self.cta_patterns):
            for match in pattern.finditer(text_lower):
                cta_matches.append(
                    {
                        "text": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern_index": i,
                        "context": text[
                            max(0, match.start() - 50) : min(
                                len(text), match.end() + 50
                            )
                        ],
                    }
                )

        # Анализ психологических триггеров в CTA
        triggers_found = {trigger: [] for trigger in self.CTA_TRIGGERS.keys()}

        for cta_match in cta_matches:
            cta_text = cta_match["text"].lower()
            for trigger_type, trigger_words in self.CTA_TRIGGERS.items():
                for trigger in trigger_words:
                    if trigger in cta_text:
                        if trigger not in triggers_found[trigger_type]:
                            triggers_found[trigger_type].append(trigger)

        # Оценка эффективности CTA по наличию триггеров
        trigger_counts = {
            trigger: len(words) for trigger, words in triggers_found.items()
        }
        total_triggers = sum(trigger_counts.values())

        # Расчет оценки эффективности CTA (от 0 до 100)
        if cta_matches:
            # Учитываем разнообразие триггеров и их количество
            trigger_diversity = len(
                [t for t, c in trigger_counts.items() if c > 0]
            ) / len(self.CTA_TRIGGERS)
            trigger_quantity = min(
                1, total_triggers / (len(cta_matches) * 2)
            )  # Не более 2 триггера на CTA

            cta_effectiveness = (trigger_diversity * 0.6 + trigger_quantity * 0.4) * 100
        else:
            cta_effectiveness = 0

        return {
            "cta_found": cta_matches,
            "cta_count": len(cta_matches),
            "triggers_found": triggers_found,
            "trigger_counts": trigger_counts,
            "cta_effectiveness": cta_effectiveness,
        }

    async def _analyze_ux_metrics(
        self, processed_text: Dict[str, Any], raw_text: str
    ) -> Dict[str, Any]:
        """
        Анализ UX-метрик текста.

        Args:
            processed_text: Предварительно обработанный текст
            raw_text: Исходный необработанный текст

        Returns:
            Словарь с UX-метриками
        """
        # Анализ ясности и доступности текста
        clarity_issues = await self._analyze_clarity(processed_text)

        # Анализ отвлекающих элементов
        distractions = await self._analyze_distractions(raw_text)

        # Анализ соответствия аудитории
        audience_metrics = await self._analyze_audience_relevance(processed_text)

        # Расчет общей оценки UX
        # Взвешенное среднее различных метрик
        clarity_score = clarity_issues.get("clarity_score", 0)
        distraction_score = 100 - distractions.get("distraction_level", 0)
        audience_score = audience_metrics.get("relevance_score", 0)

        ux_score = clarity_score * 0.4 + distraction_score * 0.3 + audience_score * 0.3

        return {
            "clarity": clarity_issues,
            "distractions": distractions,
            "audience_relevance": audience_metrics,
            "ux_score": ux_score,
        }

    async def _analyze_clarity(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализ ясности и доступности текста.

        Args:
            processed_text: Предварительно обработанный текст

        Returns:
            Словарь с метриками ясности
        """
        # Средняя длина предложения (в словах)
        sentences = processed_text.get("sentences", [])
        words = [token for token in processed_text.get("tokens", []) if token.isalpha()]

        avg_sentence_length = len(words) / len(sentences) if sentences else 0

        # Проблемы с ясностью
        clarity_issues = []

        # Слишком длинные предложения
        long_sentences = []
        for i, sentence in enumerate(sentences):
            words_in_sentence = len(sentence.split())
            if words_in_sentence > 25:  # Предложения длиннее 25 слов считаются сложными
                long_sentences.append(
                    {"index": i, "text": sentence, "word_count": words_in_sentence}
                )

        if long_sentences:
            clarity_issues.append(
                {
                    "type": "long_sentences",
                    "description": "Обнаружены слишком длинные предложения",
                    "items": long_sentences,
                }
            )

        # Расчет процента сложных слов (длиннее 6 символов)
        complex_words = [word for word in words if len(word) > 6]
        complex_word_percentage = len(complex_words) / len(words) * 100 if words else 0

        if complex_word_percentage > 30:  # Более 30% сложных слов
            clarity_issues.append(
                {
                    "type": "complex_words",
                    "description": "Высокий процент сложных слов",
                    "percentage": complex_word_percentage,
                }
            )

        # Расчет оценки ясности (от 0 до 100)
        # Чем меньше проблем, тем выше оценка
        clarity_score = 100

        # Штрафы за длинные предложения
        if avg_sentence_length > 25:
            clarity_score -= 30
        elif avg_sentence_length > 20:
            clarity_score -= 20
        elif avg_sentence_length > 15:
            clarity_score -= 10

        # Штрафы за сложные слова
        if complex_word_percentage > 30:
            clarity_score -= 30
        elif complex_word_percentage > 20:
            clarity_score -= 20
        elif complex_word_percentage > 10:
            clarity_score -= 10

        # Ограничение итоговой оценки диапазоном 0-100
        clarity_score = max(0, min(100, clarity_score))

        return {
            "avg_sentence_length": avg_sentence_length,
            "complex_word_percentage": complex_word_percentage,
            "clarity_issues": clarity_issues,
            "clarity_score": clarity_score,
        }

    async def _analyze_distractions(self, text: str) -> Dict[str, Any]:
        """
        Анализ потенциально отвлекающих элементов в тексте.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с отвлекающими элементами
        """
        # Поиск потенциально отвлекающих элементов
        distractions = []

        # Чрезмерное использование заглавных букв (CAPS)
        caps_pattern = re.compile(r"\b[A-Z][A-Z]+\b")
        caps_matches = caps_pattern.findall(text)

        if len(caps_matches) > 3:
            distractions.append(
                {
                    "type": "excessive_caps",
                    "description": f"Чрезмерное использование заглавных букв ({len(caps_matches)} случаев)",
                    "items": caps_matches[:10],  # Показываем только первые 10
                }
            )

        # Большое количество восклицательных знаков
        exclamation_count = text.count("!")
        if exclamation_count > 5:
            distractions.append(
                {
                    "type": "excessive_exclamations",
                    "description": f"Чрезмерное использование восклицательных знаков ({exclamation_count} случаев)",
                }
            )

        # Многоточия
        ellipsis_count = text.count("...")
        if ellipsis_count > 5:
            distractions.append(
                {
                    "type": "excessive_ellipsis",
                    "description": f"Чрезмерное использование многоточий ({ellipsis_count} случаев)",
                }
            )

        # Повторяющиеся фразы
        # Упрощенная проверка на повторы
        words = text.lower().split()
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)

        repeated_trigrams = [
            trigram for trigram, count in trigram_counts.items() if count > 2
        ]

        if repeated_trigrams:
            distractions.append(
                {
                    "type": "repeated_phrases",
                    "description": f"Обнаружены повторяющиеся фразы ({len(repeated_trigrams)} фраз)",
                    "items": repeated_trigrams[:5],  # Показываем только первые 5
                }
            )

        # Расчет общего уровня отвлечения (от 0 до 100)
        distraction_level = min(100, len(distractions) * 20)

        return {
            "distractions": distractions,
            "distraction_count": len(distractions),
            "distraction_level": distraction_level,
        }

    async def _analyze_audience_relevance(
        self, processed_text: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Анализ соответствия текста целевой аудитории.

        Args:
            processed_text: Предварительно обработанный текст

        Returns:
            Словарь с метриками соответствия аудитории
        """
        # Анализ соответствия целевой аудитории - более сложная задача,
        # требующая знания о целевой аудитории.
        # Здесь приведена упрощенная реализация.

        # Предполагаем, что текст должен быть доступен широкой аудитории
        tokens = processed_text.get("tokens", [])
        lemmas = processed_text.get("lemmas", [])
        sentences = processed_text.get("sentences", [])

        # Средняя длина слов (более 6 символов может быть сложно для некоторых аудиторий)
        avg_word_length = (
            sum(len(word) for word in tokens if word.isalpha()) / len(tokens)
            if tokens
            else 0
        )

        # Оценка читабельности (разнообразные метрики уже рассчитаны в BasicTextAnalyzer)
        # Тут приведем приблизительную оценку

        # Расчет схожести с целевой аудиторией (значение по умолчанию)
        relevance_score = 70  # Базовое значение

        # Корректируем оценку на основе метрик
        if avg_word_length > 6:
            relevance_score -= 20
        elif avg_word_length < 4:
            relevance_score += 10

        # Средняя длина предложения
        avg_sentence_length = len(tokens) / len(sentences) if sentences else 0

        if avg_sentence_length > 20:
            relevance_score -= 20
        elif avg_sentence_length < 10:
            relevance_score += 10

        # Ограничение итоговой оценки диапазоном 0-100
        relevance_score = max(0, min(100, relevance_score))

        return {"avg_word_length": avg_word_length, "relevance_score": relevance_score}

    async def generate_recommendations(
        self, analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Генерация рекомендаций на основе анализа тональности и UX-метрик.

        Args:
            analysis_results: Результаты анализа тональности и UX

        Returns:
            Список рекомендаций
        """
        recommendations = []

        # Рекомендации по тональности
        overall_sentiment = analysis_results.get("overall_sentiment", {})
        sentence_sentiment = analysis_results.get("sentence_sentiment", {})

        if overall_sentiment.get("sentiment") == "negative":
            recommendations.append(
                {
                    "category": "content",
                    "priority": 2,
                    "title": "Улучшите тональность текста",
                    "description": "Общая тональность текста преимущественно негативная, что может отталкивать читателей.",
                    "suggestion": "Используйте более позитивные формулировки и акцентируйте внимание на выгодах и решениях, а не на проблемах.",
                }
            )

        # Проверка на слишком нейтральный текст
        if (
            overall_sentiment.get("sentiment") == "neutral"
            and overall_sentiment.get("scores", {}).get("neutral", 0) > 0.8
        ):
            recommendations.append(
                {
                    "category": "content",
                    "priority": 3,
                    "title": "Добавьте эмоциональности",
                    "description": "Текст слишком нейтральный, что может сделать его скучным для читателей.",
                    "suggestion": "Добавьте эмоционально окрашенные слова и выражения, чтобы вызвать больший отклик у аудитории.",
                }
            )

        # Рекомендации по эмоциям
        emotion_analysis = analysis_results.get("emotion_analysis", {})

        if emotion_analysis.get("emotion_intensity", 0) < 0.05:
            recommendations.append(
                {
                    "category": "content",
                    "priority": 3,
                    "title": "Усильте эмоциональный компонент",
                    "description": "Текст имеет низкую эмоциональную интенсивность.",
                    "suggestion": "Добавьте слова и выражения, вызывающие эмоциональный отклик у читателей.",
                }
            )

        # Рекомендации по CTA
        cta_analysis = analysis_results.get("cta_analysis", {})

        if cta_analysis.get("cta_count", 0) == 0:
            recommendations.append(
                {
                    "category": "conversion",
                    "priority": 1,
                    "title": "Добавьте призывы к действию (CTA)",
                    "description": "В тексте отсутствуют явные призывы к действию.",
                    "suggestion": "Добавьте четкие призывы к действию, чтобы направить пользователей к желаемым действиям.",
                }
            )
        elif cta_analysis.get("cta_effectiveness", 0) < 40:
            recommendations.append(
                {
                    "category": "conversion",
                    "priority": 2,
                    "title": "Улучшите эффективность призывов к действию",
                    "description": f"Эффективность призывов к действию оценивается всего в {cta_analysis.get('cta_effectiveness', 0):.1f} из 100.",
                    "suggestion": "Используйте психологические триггеры (срочность, дефицит, ценность) в ваших CTA для повышения их эффективности.",
                }
            )

        # Рекомендации по UX
        ux_metrics = analysis_results.get("ux_metrics", {})
        clarity = ux_metrics.get("clarity", {})
        distractions = ux_metrics.get("distractions", {})

        if clarity.get("clarity_score", 100) < 60:
            recommendations.append(
                {
                    "category": "ux",
                    "priority": 2,
                    "title": "Повысьте ясность текста",
                    "description": f"Оценка ясности текста составляет всего {clarity.get('clarity_score', 0):.1f} из 100.",
                    "suggestion": "Упростите длинные предложения, избегайте сложных слов и используйте более простые конструкции.",
                }
            )

        if distractions.get("distraction_level", 0) > 40:
            recommendations.append(
                {
                    "category": "ux",
                    "priority": 2,
                    "title": "Уменьшите количество отвлекающих элементов",
                    "description": "В тексте обнаружено значительное количество отвлекающих элементов.",
                    "suggestion": "Уменьшите использование заглавных букв, восклицательных знаков и повторов. Сделайте текст более лаконичным.",
                }
            )

        # Рекомендации по соответствию аудитории
        audience_relevance = ux_metrics.get("audience_relevance", {})

        if audience_relevance.get("relevance_score", 100) < 60:
            recommendations.append(
                {
                    "category": "ux",
                    "priority": 2,
                    "title": "Адаптируйте текст для целевой аудитории",
                    "description": "Текст может быть недостаточно адаптирован для целевой аудитории.",
                    "suggestion": "Используйте соответствующий уровень сложности и терминологию, понятную вашей целевой аудитории.",
                }
            )

        return recommendations
