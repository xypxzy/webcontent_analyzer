import sys
import os
import json
import asyncio
from app.core.analyzer import WebContentAnalyzer, ParseOptions

# Добавляем корневую папку проекта в sys.path (если запуск идёт из другого места)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


async def analyze():
    analyzer = WebContentAnalyzer()
    options = ParseOptions(render_js=True)

    result = await analyzer.analyze_url(
        "https://newsletter.francofernando.com/p/how-not-to-fail-a-coding-interview?ref=dailydev",
        options,
    )

    # Сохраняем результат в JSON-файл
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print("Результат сохранён в result.json")


if __name__ == "__main__":
    asyncio.run(analyze())
