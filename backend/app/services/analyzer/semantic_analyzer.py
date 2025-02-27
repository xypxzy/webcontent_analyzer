import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter
import re
import math

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.tokenize import word_tokenize
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import numpy as np
from loguru import logger

# Проверка и загрузка необходимых ресурсов NLTK
try:
    import nltk
    from nltk.corpus import stopwords

    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    from nltk.corpus import stopwords


class SemanticAnalyzer:
    """
    Класс для семантического и лексического анализа текста.
    Включает извлечение ключевых слов, тематическое моделирование,
    извлечение именованных сущностей и семантические графы.
    """

    def __init__(self, lang: str = "en", models=None, config: Dict[str, Any] = None):
        """
        Инициализация семантического анализатора.

        Args:
            lang: Код языка по умолчанию ('en', 'ru', и т.д.)
            models: Загрузчик моделей
            config: Конфигурация анализатора
        """
        self.default_lang = lang
        self.models = models
        self.config = config or {}

        # Загрузка стоп-слов для поддерживаемых языков
        self.stopwords = {}
        try:
            self.stopwords["en"] = set(stopwords.words("english"))
            self.stopwords["ru"] = set(stopwords.words("russian"))
        except:
            # Если не удалось загрузить из NLTK, используем минимальный набор
            self.stopwords = {
                "en": set(
                    ["a", "an", "the", "in", "on", "at", "of", "and", "or", "to", "for"]
                ),
                "ru": set(
                    ["и", "в", "на", "с", "по", "у", "к", "о", "из", "не", "что"]
                ),
            }

        # Инициализация настроек
        self.min_keywords = self.config.get("min_keywords", 10)
        self.max_keywords = self.config.get("max_keywords", 30)
        self.n_topics = self.config.get("n_topics", 5)

    async def analyze(
        self, processed_text: Dict[str, Any], raw_text: str, lang: str
    ) -> Dict[str, Any]:
        """
        Выполнение семантического и лексического анализа текста.

        Args:
            processed_text: Предварительно обработанный текст
            raw_text: Исходный необработанный текст
            lang: Код языка

        Returns:
            Словарь с результатами семантического анализа
        """
        # Запускаем различные типы анализа параллельно
        keywords_task = asyncio.create_task(
            self._extract_keywords(processed_text, lang)
        )

        entities_task = asyncio.create_task(self._extract_entities(processed_text))

        topics_task = asyncio.create_task(
            self._extract_topics(processed_text, raw_text, lang)
        )

        semantic_graph_task = asyncio.create_task(
            self._build_semantic_graph(processed_text)
        )

        # Ожидаем завершения всех задач
        keywords = await keywords_task
        entities = await entities_task
        topics = await topics_task
        semantic_graph = await semantic_graph_task

        # Объединяем результаты
        result = {
            "keywords": keywords,
            "entities": entities,
            "topics": topics,
            "semantic_graph": semantic_graph,
            "semantic_similarity": await self._analyze_semantic_similarity(
                processed_text, lang
            ),
        }

        return result

    async def _extract_keywords(
        self, processed_text: Dict[str, Any], lang: str
    ) -> Dict[str, Any]:
        """
        Извлечение ключевых слов с помощью различных методов (TF-IDF, TextRank).

        Args:
            processed_text: Предварительно обработанный текст
            lang: Код языка

        Returns:
            Словарь с ключевыми словами и их характеристиками
        """
        # Используем лемматизированные токены без стоп-слов
        lemmas = processed_text["lemmas_without_stopwords"]

        # Если текст слишком короткий, обрабатываем особым образом
        if len(lemmas) < 20:
            return {
                "keywords": [],
                "bigrams": [],
                "density": {},
                "tfidf_keywords": [],
                "textrank_keywords": [],
            }

        # TF-IDF для одиночных слов
        tfidf_keywords = await self._extract_tfidf_keywords(processed_text, lang)

        # Поиск биграмм (словосочетаний из двух слов)
        bigrams = await self._extract_bigrams(processed_text)

        # TextRank для ключевых слов (если доступна модель spaCy)
        textrank_keywords = await self._extract_textrank_keywords(processed_text, lang)

        # Расчет плотности ключевых слов
        density = await self._calculate_keyword_density(processed_text, tfidf_keywords)

        # Объединяем результаты из разных методов и сортируем по важности
        combined_keywords = {}

        # Добавляем ключевые слова из TF-IDF
        for kw in tfidf_keywords:
            if kw["text"] not in combined_keywords:
                combined_keywords[kw["text"]] = {
                    "text": kw["text"],
                    "score": kw["score"],
                    "count": kw["count"],
                    "method": "tfidf",
                }
            else:
                combined_keywords[kw["text"]]["score"] += kw["score"]

        # Добавляем ключевые слова из TextRank
        for kw in textrank_keywords:
            if kw["text"] not in combined_keywords:
                combined_keywords[kw["text"]] = {
                    "text": kw["text"],
                    "score": kw["score"],
                    "count": kw.get("count", 1),
                    "method": "textrank",
                }
            else:
                combined_keywords[kw["text"]]["score"] += kw["score"]
                combined_keywords[kw["text"]]["method"] += "+textrank"

        # Сортируем по важности и ограничиваем количество
        keywords = sorted(
            combined_keywords.values(), key=lambda x: x["score"], reverse=True
        )[: self.max_keywords]

        return {
            "keywords": keywords,
            "bigrams": bigrams,
            "density": density,
            "tfidf_keywords": tfidf_keywords,
            "textrank_keywords": textrank_keywords,
        }

    async def _extract_tfidf_keywords(
        self, processed_text: Dict[str, Any], lang: str
    ) -> List[Dict[str, Any]]:
        """
        Извлечение ключевых слов с помощью TF-IDF.

        Args:
            processed_text: Предварительно обработанный текст
            lang: Код языка

        Returns:
            Список словарей с ключевыми словами и их значениями TF-IDF
        """
        # Используем лемматизированные токены без стоп-слов
        lemmas = processed_text["lemmas_without_stopwords"]

        # Подготавливаем документы для TF-IDF (в данном случае один документ)
        documents = [" ".join(lemmas)]

        # Создаем TF-IDF векторизатор
        try:
            loop = asyncio.get_event_loop()

            # TF-IDF векторизация выполняется в отдельном потоке
            vectorizer = TfidfVectorizer(
                max_features=100,  # Ограничиваем количество признаков
                min_df=1,
                max_df=0.9,
            )

            # Векторизация текста
            tfidf_matrix = await loop.run_in_executor(
                None, vectorizer.fit_transform, documents
            )

            # Получаем слова и их TF-IDF значения
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Создаем словарь "слово: TF-IDF" и сортируем по убыванию
            word_scores = [
                (feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))
            ]
            word_scores.sort(key=lambda x: x[1], reverse=True)

            # Считаем частоту каждого слова в тексте
            word_counts = Counter(lemmas)

            # Формируем результаты
            keywords = []
            for word, score in word_scores[: self.max_keywords]:
                keywords.append(
                    {
                        "text": word,
                        "score": float(score),
                        "count": word_counts.get(word, 0),
                    }
                )

            return keywords

        except Exception as e:
            logger.error(f"Ошибка при извлечении ключевых слов TF-IDF: {str(e)}")
            return []

    async def _extract_textrank_keywords(
        self, processed_text: Dict[str, Any], lang: str
    ) -> List[Dict[str, Any]]:
        """
        Извлечение ключевых слов с помощью TextRank.

        Args:
            processed_text: Предварительно обработанный текст
            lang: Код языка

        Returns:
            Список словарей с ключевыми словами и их значениями по TextRank
        """
        # Используем готовый объект Doc из spaCy
        doc = processed_text.get("doc")

        # Если нет объекта Doc или длина текста слишком мала
        if not doc or len(doc) < 20:
            return []

        try:
            # TextRank реализован непосредственно в spaCy
            # для некоторых моделей через расширение
            if hasattr(doc, "_.textrank"):
                # Если есть расширение TextRank
                textrank_keywords = [
                    {"text": kw[0], "score": kw[1]}
                    for kw in doc._.textrank.sort_by_score()[: self.max_keywords]
                ]
            else:
                # Используем наивную реализацию на основе POS тегов
                # Выбираем только существительные и прилагательные
                candidates = []
                for token in doc:
                    if token.pos_ in {"NOUN", "PROPN", "ADJ"} and not token.is_stop:
                        candidates.append(token.lemma_)

                # Считаем частоту кандидатов
                counts = Counter(candidates)

                # Нормализуем счетчик, чтобы получить "значения TextRank"
                total_count = sum(counts.values())
                normalized_counts = {
                    word: count / total_count for word, count in counts.items()
                }

                # Сортируем по частоте и ограничиваем количество
                sorted_keywords = sorted(
                    normalized_counts.items(), key=lambda x: x[1], reverse=True
                )[: self.max_keywords]

                textrank_keywords = [
                    {"text": kw[0], "score": kw[1], "count": counts[kw[0]]}
                    for kw in sorted_keywords
                ]

            return textrank_keywords

        except Exception as e:
            logger.error(f"Ошибка при извлечении ключевых слов TextRank: {str(e)}")
            return []

    async def _extract_bigrams(
        self, processed_text: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Извлечение значимых биграмм (словосочетаний из двух слов).

        Args:
            processed_text: Предварительно обработанный текст

        Returns:
            Список биграмм с их характеристиками
        """
        # Используем лемматизированные токены без стоп-слов
        lemmas = processed_text["lemmas_without_stopwords"]

        # Если текст слишком короткий, биграммы не имеют смысла
        if len(lemmas) < 20:
            return []

        try:
            # Используем Gensim Phrases для обнаружения биграмм
            loop = asyncio.get_event_loop()

            # Подготавливаем текст как список токенов (один документ)
            texts = [lemmas]

            # Построение модели биграмм
            phrases = await loop.run_in_executor(
                None, Phrases, texts, min_count=3, threshold=10
            )
            bigram = await loop.run_in_executor(None, Phraser, phrases)

            # Получаем биграммы из текста
            bigram_tokens = await loop.run_in_executor(None, bigram, lemmas)

            # Фильтруем только биграммы (содержат "_")
            bigrams_only = [token for token in bigram_tokens if "_" in token]

            # Считаем частоту биграмм
            bigram_counts = Counter(bigrams_only)

            # Формируем результаты
            result_bigrams = []
            for bigram, count in bigram_counts.most_common(self.max_keywords // 2):
                words = bigram.split("_")
                result_bigrams.append(
                    {
                        "text": " ".join(words),
                        "count": count,
                        "score": count / len(lemmas) * 100,  # Относительная частота
                        "words": words,
                    }
                )

            return result_bigrams

        except Exception as e:
            logger.error(f"Ошибка при извлечении биграмм: {str(e)}")
            return []

    async def _calculate_keyword_density(
        self, processed_text: Dict[str, Any], keywords: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Расчет плотности ключевых слов (общая и по секциям).

        Args:
            processed_text: Предварительно обработанный текст
            keywords: Список ключевых слов

        Returns:
            Словарь с плотностью ключевых слов
        """
        # Получаем токены и общее количество слов
        tokens = processed_text["tokens"]
        total_words = len([t for t in tokens if t.isalpha()])

        if total_words == 0:
            return {"overall": 0, "top_keywords": [], "sections": []}

        # Расчет общей плотности ключевых слов
        keyword_count = sum(kw["count"] for kw in keywords if "count" in kw)
        overall_density = keyword_count / total_words * 100 if total_words > 0 else 0

        # Топ ключевых слов по плотности
        top_keywords = []
        for kw in keywords[:10]:  # Берем только первые 10 ключевых слов
            if "count" in kw:
                density = kw["count"] / total_words * 100
                top_keywords.append(
                    {"text": kw["text"], "count": kw["count"], "density": density}
                )

        # Плотность по секциям (разбиваем текст на примерно равные части)
        sections = []
        section_size = max(
            100, total_words // 5
        )  # Примерно 5 секций, но не менее 100 слов

        # В реальной системе здесь был бы более сложный алгоритм разбивки на секции
        # на основе структуры HTML, заголовков и т.д.

        current_section = []
        current_section_word_count = 0
        section_index = 0

        for token in tokens:
            if token.isalpha():
                current_section.append(token)
                current_section_word_count += 1

                if current_section_word_count >= section_size:
                    # Анализируем секцию
                    section_keywords = []
                    section_tokens_text = [t.lower() for t in current_section]

                    for kw in keywords[
                        :5
                    ]:  # Берем только топ-5 ключевых слов для каждой секции
                        kw_count = section_tokens_text.count(kw["text"].lower())
                        if kw_count > 0:
                            section_keywords.append(
                                {
                                    "text": kw["text"],
                                    "count": kw_count,
                                    "density": kw_count
                                    / current_section_word_count
                                    * 100,
                                }
                            )

                    sections.append(
                        {
                            "index": section_index,
                            "word_count": current_section_word_count,
                            "keywords": section_keywords,
                        }
                    )

                    # Сбрасываем счетчики для следующей секции
                    current_section = []
                    current_section_word_count = 0
                    section_index += 1

        # Добавляем последнюю секцию, если она не пустая
        if current_section_word_count > 0:
            section_keywords = []
            section_tokens_text = [t.lower() for t in current_section]

            for kw in keywords[:5]:
                kw_count = section_tokens_text.count(kw["text"].lower())
                if kw_count > 0:
                    section_keywords.append(
                        {
                            "text": kw["text"],
                            "count": kw_count,
                            "density": kw_count / current_section_word_count * 100,
                        }
                    )

            sections.append(
                {
                    "index": section_index,
                    "word_count": current_section_word_count,
                    "keywords": section_keywords,
                }
            )

        return {
            "overall": overall_density,
            "top_keywords": top_keywords,
            "sections": sections,
        }

    async def _extract_entities(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлечение и классификация именованных сущностей.

        Args:
            processed_text: Предварительно обработанный текст

        Returns:
            Словарь с именованными сущностями по категориям
        """
        # Используем результат NER из spaCy, полученный во время предобработки
        entities = processed_text.get("entities", [])

        # Группируем сущности по типам
        entity_types = {}
        for entity in entities:
            entity_type = entity["label"]
            if entity_type not in entity_types:
                entity_types[entity_type] = []

            entity_types[entity_type].append(entity)

        # Выявление наиболее важных сущностей
        all_entities = []
        for entity_type, entities_list in entity_types.items():
            # Считаем частоту каждой сущности
            entity_counts = {}
            for entity in entities_list:
                text = entity["text"].lower()
                if text not in entity_counts:
                    entity_counts[text] = {
                        "text": entity["text"],
                        "count": 0,
                        "type": entity_type,
                    }
                entity_counts[text]["count"] += 1

            # Сортируем по частоте и добавляем в общий список
            sorted_entities = sorted(
                entity_counts.values(), key=lambda x: x["count"], reverse=True
            )

            all_entities.extend(sorted_entities)

        # Сортируем все сущности по частоте
        all_entities.sort(key=lambda x: x["count"], reverse=True)

        return {
            "by_type": entity_types,
            "all_entities": all_entities[:50],  # Ограничиваем общее количество
        }

    async def _extract_topics(
        self, processed_text: Dict[str, Any], raw_text: str, lang: str
    ) -> Dict[str, Any]:
        """
        Выполнение тематического моделирования (LDA, NMF).

        Args:
            processed_text: Предварительно обработанный текст
            raw_text: Исходный необработанный текст
            lang: Код языка

        Returns:
            Словарь с выделенными темами и их характеристиками
        """
        # Если текст слишком короткий, темы не имеют смысла
        if len(processed_text["lemmas_without_stopwords"]) < 50:
            return {"topics": [], "main_topic": None, "topic_distribution": []}

        try:
            # Подготовка данных для тематического моделирования
            lemmas = processed_text["lemmas_without_stopwords"]
            text = " ".join(lemmas)

            # Векторизация текста с помощью TF-IDF
            loop = asyncio.get_event_loop()

            vectorizer = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.95)

            # Векторизация текста
            tfidf_matrix = await loop.run_in_executor(
                None, vectorizer.fit_transform, [text]
            )
            feature_names = vectorizer.get_feature_names_out()

            # Создаем модель NMF (Non-negative Matrix Factorization)
            n_topics = min(self.n_topics, 3)  # Для одного документа не более 3 тем
            nmf_model = await loop.run_in_executor(
                None, lambda: NMF(n_components=n_topics, random_state=42, max_iter=1000)
            )

            # Обучаем модель NMF
            nmf_weights = await loop.run_in_executor(
                None, nmf_model.fit_transform, tfidf_matrix
            )

            # Извлекаем темы
            topics = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words_idx = topic.argsort()[:-11:-1]  # 10 топ слов
                top_words = [feature_names[i] for i in top_words_idx]
                topic_weight = float(nmf_weights[0][topic_idx])

                topics.append(
                    {
                        "id": topic_idx,
                        "weight": topic_weight,
                        "top_words": top_words,
                        "label": f"Тема {topic_idx + 1}",  # В реальной системе тут было бы название
                    }
                )

            # Сортируем темы по весу
            topics.sort(key=lambda x: x["weight"], reverse=True)

            # Определяем основную тему
            main_topic = topics[0] if topics else None

            # Распределение тем
            topic_distribution = []
            total_weight = sum(topic["weight"] for topic in topics)

            if total_weight > 0:
                for topic in topics:
                    percentage = topic["weight"] / total_weight * 100
                    topic_distribution.append(
                        {
                            "id": topic["id"],
                            "label": topic["label"],
                            "percentage": percentage,
                        }
                    )

            return {
                "topics": topics,
                "main_topic": main_topic,
                "topic_distribution": topic_distribution,
            }

        except Exception as e:
            logger.error(f"Ошибка при выполнении тематического моделирования: {str(e)}")
            return {
                "topics": [],
                "main_topic": None,
                "topic_distribution": [],
                "error": str(e),
            }

    async def _build_semantic_graph(
        self, processed_text: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Построение семантического графа связей между ключевыми сущностями.

        Args:
            processed_text: Предварительно обработанный текст

        Returns:
            Словарь с семантическим графом (узлы и связи)
        """
        # Для построения полноценного семантического графа нужны более сложные алгоритмы
        # Здесь приведена упрощенная версия

        # Используем именованные сущности и зависимости из spaCy
        entities = processed_text.get("entities", [])
        doc = processed_text.get("doc")

        if not doc or len(entities) < 2:
            return {"nodes": [], "edges": []}

        try:
            # Создаем узлы из именованных сущностей
            nodes = []
            node_map = {}  # для быстрого поиска узла по тексту

            for i, entity in enumerate(entities):
                # Используем только уникальные сущности
                if entity["text"].lower() not in node_map:
                    node = {
                        "id": i,
                        "text": entity["text"],
                        "type": entity["label"],
                        "weight": 1,  # начальный вес
                    }
                    nodes.append(node)
                    node_map[entity["text"].lower()] = i
                else:
                    # Если сущность уже есть, увеличиваем ее вес
                    node_id = node_map[entity["text"].lower()]
                    nodes[node_id]["weight"] += 1

            # Создаем связи между сущностями, которые встречаются в одном предложении
            edges = []
            edge_set = set()  # для предотвращения дублирования связей

            for sent in doc.sents:
                # Сущности в текущем предложении
                sentence_entities = []
                for ent in entities:
                    if sent.start_char <= ent["start"] < sent.end_char:
                        if ent["text"].lower() in node_map:
                            sentence_entities.append(ent)

                # Создаем связи между всеми сущностями в предложении
                for i, ent1 in enumerate(sentence_entities):
                    for ent2 in sentence_entities[i + 1 :]:
                        node1_id = node_map[ent1["text"].lower()]
                        node2_id = node_map[ent2["text"].lower()]

                        # Создаем уникальный идентификатор связи
                        edge_id = tuple(sorted([node1_id, node2_id]))

                        if edge_id not in edge_set:
                            edges.append(
                                {"source": node1_id, "target": node2_id, "weight": 1}
                            )
                            edge_set.add(edge_id)
                        else:
                            # Если связь уже существует, увеличиваем ее вес
                            for edge in edges:
                                if (
                                    edge["source"] == node1_id
                                    and edge["target"] == node2_id
                                ) or (
                                    edge["source"] == node2_id
                                    and edge["target"] == node1_id
                                ):
                                    edge["weight"] += 1
                                    break

            return {"nodes": nodes, "edges": edges}

        except Exception as e:
            logger.error(f"Ошибка при построении семантического графа: {str(e)}")
            return {"nodes": [], "edges": [], "error": str(e)}

    async def _analyze_semantic_similarity(
        self, processed_text: Dict[str, Any], lang: str
    ) -> Dict[str, Any]:
        """
        Анализ семантической согласованности и связности текста.

        Args:
            processed_text: Предварительно обработанный текст
            lang: Код языка

        Returns:
            Словарь с метриками семантической близости
        """
        # Для анализа семантической согласованности и связности
        # в реальной системе использовались бы модели семантических эмбеддингов
        # и алгоритмы анализа когезии и когерентности

        # Для примера используем упрощенную модель
        doc = processed_text.get("doc")
        sentences = processed_text.get("sentences", [])

        if not doc or len(sentences) < 2:
            return {"coherence_score": 1.0, "sentence_similarity": []}

        try:
            # В реальной системе тут использовался бы анализ с помощью моделей трансформеров
            # Но для примера используем упрощенный подход на основе словарного пересечения

            # Преобразуем предложения в множества слов
            sentence_tokens = []
            for sent in doc.sents:
                # Берем только значимые слова (не стоп-слова)
                tokens = [
                    token.lemma_
                    for token in sent
                    if not token.is_stop and token.is_alpha
                ]
                sentence_tokens.append(tokens)

            # Рассчитываем сходство между соседними предложениями
            sentence_similarity = []

            for i in range(len(sentence_tokens) - 1):
                set1 = set(sentence_tokens[i])
                set2 = set(sentence_tokens[i + 1])

                # Коэффициент Жаккара для измерения сходства
                if len(set1) > 0 and len(set2) > 0:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = intersection / union if union > 0 else 0
                else:
                    similarity = 0

                sentence_similarity.append(
                    {
                        "first_sentence_index": i,
                        "second_sentence_index": i + 1,
                        "similarity": similarity,
                    }
                )

            # Средняя согласованность текста
            avg_coherence = (
                sum(s["similarity"] for s in sentence_similarity)
                / len(sentence_similarity)
                if sentence_similarity
                else 0
            )

            return {
                "coherence_score": avg_coherence,
                "sentence_similarity": sentence_similarity,
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе семантической согласованности: {str(e)}")
            return {"coherence_score": 0.5, "error": str(e)}  # Значение по умолчанию

    async def generate_recommendations(
        self, analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Генерация рекомендаций на основе семантического анализа.

        Args:
            analysis_results: Результаты семантического анализа

        Returns:
            Список рекомендаций
        """
        recommendations = []

        # Рекомендации по ключевым словам
        keywords = analysis_results.get("keywords", {}).get("keywords", [])
        density = analysis_results.get("keywords", {}).get("density", {})

        if keywords:
            # Проверка плотности ключевых слов
            overall_density = density.get("overall", 0)

            if overall_density < 1.0:
                recommendations.append(
                    {
                        "category": "seo",
                        "priority": 2,
                        "title": "Увеличьте плотность ключевых слов",
                        "description": f"Общая плотность ключевых слов ({overall_density:.1f}%) слишком низкая.",
                        "suggestion": f"Добавьте больше ключевых слов в текст. Оптимальная плотность - 1.5-2.5%. Рекомендуемые ключевые слова: {', '.join(kw['text'] for kw in keywords[:5])}",
                    }
                )
            elif overall_density > 5.0:
                recommendations.append(
                    {
                        "category": "seo",
                        "priority": 2,
                        "title": "Уменьшите плотность ключевых слов",
                        "description": f"Общая плотность ключевых слов ({overall_density:.1f}%) слишком высокая, что может быть расценено как переоптимизация.",
                        "suggestion": "Снизьте частоту использования ключевых слов и добавьте больше уникального контента.",
                    }
                )

            # Проверка равномерности распределения ключевых слов
            sections = density.get("sections", [])
            if sections and len(sections) > 1:
                section_densities = [
                    sum(kw["density"] for kw in section.get("keywords", []))
                    for section in sections
                ]

                max_density = max(section_densities) if section_densities else 0
                min_density = min(section_densities) if section_densities else 0

                if max_density > 0 and min_density / max_density < 0.3:
                    recommendations.append(
                        {
                            "category": "seo",
                            "priority": 3,
                            "title": "Распределите ключевые слова равномернее",
                            "description": "Ключевые слова сконцентрированы только в некоторых частях текста.",
                            "suggestion": "Распределите ключевые слова более равномерно по всему тексту для улучшения SEO.",
                        }
                    )

        # Рекомендации по семантической согласованности
        coherence = analysis_results.get("semantic_similarity", {}).get(
            "coherence_score", 0
        )

        if coherence < 0.2:
            recommendations.append(
                {
                    "category": "content",
                    "priority": 2,
                    "title": "Улучшите связность текста",
                    "description": "Текст имеет низкую семантическую связность между предложениями.",
                    "suggestion": "Добавьте переходные фразы и логические связки между предложениями. Убедитесь, что каждое предложение логически связано с предыдущим и следующим.",
                }
            )

        # Рекомендации по тематике
        topics = analysis_results.get("topics", {}).get("topics", [])

        if topics and len(topics) > 1:
            # Если есть несколько тем с близкими весами, это может указывать на размытость темы
            top_weights = [topic["weight"] for topic in topics[:2]]

            if len(top_weights) > 1 and top_weights[0] / top_weights[1] < 1.5:
                recommendations.append(
                    {
                        "category": "content",
                        "priority": 2,
                        "title": "Уточните основную тему",
                        "description": "Текст содержит несколько примерно равных по важности тем, что может размывать фокус.",
                        "suggestion": f"Сосредоточьтесь на одной основной теме. Рекомендуемые ключевые слова для основной темы: {', '.join(topics[0]['top_words'][:5])}",
                    }
                )

        # Рекомендации по именованным сущностям
        entities = analysis_results.get("entities", {}).get("all_entities", [])

        if not entities:
            recommendations.append(
                {
                    "category": "content",
                    "priority": 3,
                    "title": "Добавьте именованные сущности",
                    "description": "В тексте мало или отсутствуют именованные сущности (имена, организации, локации).",
                    "suggestion": "Добавьте конкретные примеры, имена компаний, персон, местоположений для увеличения релевантности и информативности.",
                }
            )

        return recommendations
