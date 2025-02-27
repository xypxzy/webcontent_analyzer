import asyncio
from typing import Dict, List, Any, Optional
import re

# import math

import textstat
from loguru import logger
import language_tool_python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from app.services.analyzer.models.model_loader import ModelLoader


class BasicTextAnalyzer:
    """
    Класс для базового анализа текста: подсчет статистик,
    анализ удобочитаемости, грамматические проверки и т.д.
    """

    def __init__(self, lang: str = "en", models: Optional[ModelLoader] = None):
        """
        Инициализация анализатора.

        Args:
            lang: Код языка по умолчанию ('en', 'ru', и т.д.)
            models: Загрузчик моделей (если None, будет создан новый)
        """
        self.default_lang = lang
        self.models = models if models else ModelLoader()

        # Инициализация инструмента проверки грамматики
        # Для продакшена лучше использовать настраиваемый сервер,
        # но для примера используем локальную версию
        try:
            self.grammar_tool = language_tool_python.LanguageTool(self.default_lang)
            logger.info(
                f"Инициализирован инструмент проверки грамматики для языка {self.default_lang}"
            )
        except Exception as e:
            logger.error(f"Не удалось инициализировать LanguageTool: {e}")
            self.grammar_tool = None

    async def analyze(
        self, processed_text: Dict[str, Any], lang: str
    ) -> Dict[str, Any]:
        """
        Выполнение базового анализа текста.

        Args:
            processed_text: Предварительно обработанный текст
            lang: Код языка

        Returns:
            Словарь с результатами базового анализа
        """
        # Выполняем различные типы анализа параллельно
        text_stats_task = asyncio.create_task(
            self._calculate_text_statistics(processed_text)
        )

        readability_task = asyncio.create_task(
            self._calculate_readability_metrics(processed_text, lang)
        )

        grammar_task = asyncio.create_task(
            self._check_grammar(processed_text["normalized_text"], lang)
        )

        # Ожидаем завершения всех задач
        text_stats = await text_stats_task
        readability_metrics = await readability_task
        grammar_issues = await grammar_task

        # Объединяем результаты
        result = {
            "text_statistics": text_stats,
            "readability": readability_metrics,
            "grammar_issues": grammar_issues,
            "overall_quality_score": self._calculate_overall_quality(
                text_stats, readability_metrics, grammar_issues
            ),
        }

        return result

    async def _calculate_text_statistics(
        self, processed_text: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Расчет базовых статистических метрик текста.

        Args:
            processed_text: Предварительно обработанный текст

        Returns:
            Словарь со статистическими метриками
        """
        # Извлекаем необходимые данные из предобработанного текста
        tokens = processed_text["tokens"]
        sentences = processed_text["sentences"]
        words = [token for token in tokens if token.isalpha()]

        # Расчет параграфов (примерно, по двойным переносам строк)
        paragraphs = re.split(r"\n\s*\n", processed_text["original_text"])
        paragraphs = [p for p in paragraphs if p.strip()]

        # Расчет лексического разнообразия
        if len(words) > 0:
            unique_words = set(word.lower() for word in words)
            lexical_diversity = len(unique_words) / len(words)
        else:
            lexical_diversity = 0

        # Расчет средней длины предложения (в словах)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0

        # Расчет средней длины слова (в символах)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # Расчет плотности пунктуации
        punctuation_count = sum(1 for token in tokens if token in ".,;:!?()[]{}\"'")
        punctuation_density = punctuation_count / len(tokens) if tokens else 0

        # Расчет количества чисел
        numbers_count = sum(1 for token in tokens if token.isdigit())

        # Расчет длинных слов (более 6 символов)
        long_words_count = sum(1 for word in words if len(word) > 6)
        long_words_percentage = long_words_count / len(words) * 100 if words else 0

        return {
            "word_count": len(words),
            "character_count": sum(len(word) for word in words),
            "character_count_with_spaces": len(processed_text["normalized_text"]),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "lexical_diversity": lexical_diversity,
            "punctuation_count": punctuation_count,
            "punctuation_density": punctuation_density,
            "numbers_count": numbers_count,
            "long_words_count": long_words_count,
            "long_words_percentage": long_words_percentage,
        }

    async def _calculate_readability_metrics(
        self, processed_text: Dict[str, Any], lang: str
    ) -> Dict[str, Any]:
        """
        Расчет метрик удобочитаемости текста.

        Args:
            processed_text: Предварительно обработанный текст
            lang: Код языка

        Returns:
            Словарь с метриками удобочитаемости
        """
        # Для индексов удобочитаемости используем библиотеку textstat
        # Но учитываем, что она лучше всего работает с английским языком

        text = processed_text["normalized_text"]

        # Определяем, какие метрики доступны для данного языка
        metrics = {}

        # Установка языка для textstat
        if lang == "en":
            textstat.set_lang("en")

            # Расчет различных индексов удобочитаемости
            metrics["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
            metrics["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
            metrics["smog_index"] = textstat.smog_index(text)
            metrics["coleman_liau_index"] = textstat.coleman_liau_index(text)
            metrics["automated_readability_index"] = (
                textstat.automated_readability_index(text)
            )
            metrics["dale_chall_readability_score"] = (
                textstat.dale_chall_readability_score(text)
            )
            metrics["difficult_words"] = textstat.difficult_words(text)
            metrics["linsear_write_formula"] = textstat.linsear_write_formula(text)
            metrics["gunning_fog"] = textstat.gunning_fog(text)

            # Интерпретация индекса Flesch Reading Ease
            if metrics["flesch_reading_ease"] >= 90:
                metrics["flesch_reading_ease_description"] = (
                    "Очень легко читаемый текст"
                )
                metrics["education_level"] = "5 класс"
            elif metrics["flesch_reading_ease"] >= 80:
                metrics["flesch_reading_ease_description"] = "Легко читаемый текст"
                metrics["education_level"] = "6 класс"
            elif metrics["flesch_reading_ease"] >= 70:
                metrics["flesch_reading_ease_description"] = (
                    "Довольно легко читаемый текст"
                )
                metrics["education_level"] = "7 класс"
            elif metrics["flesch_reading_ease"] >= 60:
                metrics["flesch_reading_ease_description"] = "Обычный/разговорный текст"
                metrics["education_level"] = "8-9 класс"
            elif metrics["flesch_reading_ease"] >= 50:
                metrics["flesch_reading_ease_description"] = "Умеренно сложный текст"
                metrics["education_level"] = "10-12 класс"
            elif metrics["flesch_reading_ease"] >= 30:
                metrics["flesch_reading_ease_description"] = "Сложный текст"
                metrics["education_level"] = "Колледж"
            else:
                metrics["flesch_reading_ease_description"] = "Очень сложный текст"
                metrics["education_level"] = "Высшее образование"

        elif lang == "ru":
            # Для русского языка используем модифицированную формулу Flesch Reading Ease
            # Формула: 206.835 - 1.3 * avg_sentence_length - 60.1 * avg_syllables_per_word

            # Подсчет слогов для русских слов (приблизительно)
            def count_syllables_ru(word):
                vowels = "аеёиоуыэюя"
                count = sum(1 for char in word.lower() if char in vowels)
                return max(1, count)  # Минимум 1 слог

            words = [token for token in processed_text["tokens"] if token.isalpha()]
            sentences = processed_text["sentences"]

            if words and sentences:
                avg_sentence_length = len(words) / len(sentences)
                syllables = [count_syllables_ru(word) for word in words]
                avg_syllables_per_word = sum(syllables) / len(words)

                # Адаптированная формула Flesch Reading Ease для русского
                flesch_ru = (
                    206.835
                    - (1.3 * avg_sentence_length)
                    - (60.1 * avg_syllables_per_word)
                )
                metrics["flesch_reading_ease_ru"] = max(0, min(100, flesch_ru))

                # Приблизительный индекс Gunning Fog для русского
                complex_words = sum(1 for s in syllables if s >= 3)
                complex_words_percent = complex_words / len(words) * 100
                gunning_fog_ru = 0.4 * (avg_sentence_length + complex_words_percent)
                metrics["gunning_fog_ru"] = gunning_fog_ru
            else:
                metrics["flesch_reading_ease_ru"] = 0
                metrics["gunning_fog_ru"] = 0

        # Добавляем общую оценку удобочитаемости
        if "flesch_reading_ease" in metrics:
            readability_score = metrics["flesch_reading_ease"]
        elif "flesch_reading_ease_ru" in metrics:
            readability_score = metrics["flesch_reading_ease_ru"]
        else:
            readability_score = 50  # Значение по умолчанию

        metrics["overall_readability_score"] = readability_score

        return metrics

    async def _check_grammar(self, text: str, lang: str) -> Dict[str, Any]:
        """
        Проверка грамматики и пунктуации.

        Args:
            text: Текст для проверки
            lang: Код языка

        Returns:
            Словарь с найденными грамматическими ошибками
        """
        # Если LanguageTool не инициализирован, возвращаем пустой результат
        if not self.grammar_tool:
            return {
                "error_count": 0,
                "errors": [],
                "grammar_score": 100,  # По умолчанию идеальная оценка
            }

        try:
            # Запускаем проверку грамматики в отдельном потоке, так как это блокирующая операция
            loop = asyncio.get_event_loop()

            # Проверка грамматики для первых 10000 символов (для производительности)
            text_to_check = text[:10000]
            matches = await loop.run_in_executor(
                None, self.grammar_tool.check, text_to_check
            )

            # Преобразуем результаты в удобный формат
            errors = []
            for match in matches:
                errors.append(
                    {
                        "message": match.message,
                        "context": match.context,
                        "offset": match.offset,
                        "length": match.errorLength,
                        "rule_id": match.ruleId,
                        "category": match.category,
                        "replacements": match.replacements[
                            :5
                        ],  # Ограничиваем количество замен
                    }
                )

            # Расчет оценки грамматики
            # Чем меньше ошибок на 100 слов, тем выше оценка
            words_count = len(text_to_check.split())
            error_rate = len(errors) / words_count * 100 if words_count > 0 else 0
            grammar_score = max(
                0, 100 - error_rate * 10
            )  # 10 ошибок на 100 слов = 0 баллов

            return {
                "error_count": len(errors),
                "errors": errors,
                "grammar_score": grammar_score,
            }
        except Exception as e:
            logger.error(f"Ошибка при проверке грамматики: {e}")
            return {
                "error_count": 0,
                "errors": [],
                "grammar_score": 50,  # Средняя оценка при ошибке
                "error_message": str(e),
            }

    def _calculate_overall_quality(
        self,
        text_stats: Dict[str, Any],
        readability_metrics: Dict[str, Any],
        grammar_issues: Dict[str, Any],
    ) -> float:
        """
        Расчет общей оценки качества текста на основе всех метрик.

        Args:
            text_stats: Статистика текста
            readability_metrics: Метрики удобочитаемости
            grammar_issues: Найденные грамматические ошибки

        Returns:
            Общая оценка качества текста (0-100)
        """
        # Берем ключевые метрики из разных категорий
        readability_score = readability_metrics.get("overall_readability_score", 50)
        lexical_diversity = (
            text_stats.get("lexical_diversity", 0.5) * 100
        )  # Переводим в шкалу 0-100
        grammar_score = grammar_issues.get("grammar_score", 50)

        # Взвешенный расчет общей оценки
        weights = {"readability": 0.4, "lexical_diversity": 0.3, "grammar": 0.3}

        overall_score = (
            readability_score * weights["readability"]
            + lexical_diversity * weights["lexical_diversity"]
            + grammar_score * weights["grammar"]
        )

        # Ограничиваем значение диапазоном 0-100
        return max(0, min(100, overall_score))

    async def generate_recommendations(
        self, analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Генерация рекомендаций по улучшению текста на основе базового анализа.

        Args:
            analysis_results: Результаты базового анализа

        Returns:
            Список рекомендаций
        """
        recommendations = []
        text_stats = analysis_results["text_statistics"]
        readability = analysis_results["readability"]
        grammar_issues = analysis_results["grammar_issues"]

        # Рекомендации по длине предложений
        avg_sentence_length = text_stats.get("avg_sentence_length", 0)
        if avg_sentence_length > 25:
            recommendations.append(
                {
                    "category": "content",
                    "priority": 2,
                    "title": "Упростите предложения",
                    "description": f"Средняя длина предложения ({avg_sentence_length:.1f} слов) слишком велика. "
                    + "Длинные предложения затрудняют чтение и понимание текста.",
                    "suggestion": "Разделите длинные предложения на более короткие. Стремитесь к средней длине предложения в 15-20 слов.",
                }
            )

        # Рекомендации по удобочитаемости
        if (
            "flesch_reading_ease" in readability
            and readability["flesch_reading_ease"] < 50
        ):
            recommendations.append(
                {
                    "category": "content",
                    "priority": 2,
                    "title": "Повысьте удобочитаемость текста",
                    "description": f"Индекс удобочитаемости Flesch ({readability['flesch_reading_ease']:.1f}) указывает на то, "
                    + "что текст может быть слишком сложным для восприятия целевой аудиторией.",
                    "suggestion": "Используйте более простые слова, сократите длину предложений и избегайте сложных конструкций.",
                }
            )
        elif (
            "flesch_reading_ease_ru" in readability
            and readability["flesch_reading_ease_ru"] < 50
        ):
            recommendations.append(
                {
                    "category": "content",
                    "priority": 2,
                    "title": "Повысьте удобочитаемость текста",
                    "description": f"Индекс удобочитаемости ({readability['flesch_reading_ease_ru']:.1f}) указывает на то, "
                    + "что текст может быть слишком сложным для восприятия целевой аудиторией.",
                    "suggestion": "Используйте более простые слова, сократите длину предложений и избегайте сложных конструкций.",
                }
            )

        # Рекомендации по лексическому разнообразию
        lexical_diversity = text_stats.get("lexical_diversity", 0)
        if lexical_diversity < 0.4:
            recommendations.append(
                {
                    "category": "content",
                    "priority": 3,
                    "title": "Увеличьте лексическое разнообразие",
                    "description": f"Низкий показатель лексического разнообразия ({lexical_diversity:.2f}) указывает на частое повторение одних и тех же слов.",
                    "suggestion": "Используйте синонимы и альтернативные формулировки, чтобы избежать повторений.",
                }
            )
        elif lexical_diversity > 0.8 and text_stats.get("word_count", 0) > 300:
            recommendations.append(
                {
                    "category": "content",
                    "priority": 4,
                    "title": "Обеспечьте терминологическую согласованность",
                    "description": f"Очень высокий показатель лексического разнообразия ({lexical_diversity:.2f}) может указывать на непоследовательное использование терминологии.",
                    "suggestion": "Удостоверьтесь, что используете одни и те же термины для одних и тех же понятий.",
                }
            )

        # Рекомендации по грамматике
        error_count = grammar_issues.get("error_count", 0)
        if error_count > 5:
            recommendations.append(
                {
                    "category": "content",
                    "priority": 1,
                    "title": "Исправьте грамматические ошибки",
                    "description": f"В тексте обнаружено {error_count} грамматических или пунктуационных ошибок.",
                    "suggestion": "Проверьте текст с помощью программы проверки грамматики или обратитесь к редактору.",
                }
            )

        # Рекомендации по структуре текста
        paragraph_count = text_stats.get("paragraph_count", 0)
        sentence_count = text_stats.get("sentence_count", 0)

        if paragraph_count > 0 and sentence_count / paragraph_count > 7:
            recommendations.append(
                {
                    "category": "structure",
                    "priority": 3,
                    "title": "Улучшите структуру абзацев",
                    "description": f"Средний абзац содержит слишком много предложений ({sentence_count / paragraph_count:.1f}).",
                    "suggestion": "Разбейте длинные абзацы на более короткие для улучшения визуального восприятия и читаемости.",
                }
            )

        return recommendations
