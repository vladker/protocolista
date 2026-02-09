#!/usr/bin/env python3
"""
Генерация саммари с помощью Ollama + Gemma
"""

import argparse
import json
import sys
import os
from pathlib import Path
import requests


def load_tagged_json(json_path):
    """Загрузка тегированного JSON"""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def format_transcript(tagged_segments, max_chars=15000):
    """Форматирование транскрипции для отправки в LLM"""
    result = []
    for seg in tagged_segments:
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        result.append(f"[{speaker}]: {text}")
    
    full_text = "\n".join(result)
    
    # Ограничение длины
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n... (сокращено)"
    
    return full_text


def generate_summary(text, model="gemma3:27b", timeout=120):
    """Генерация саммари через Ollama API"""
    
    system_prompt = """Ты мой эффективный AI-ассистент по анализу стенограмм совещаний и лекций.

Сделай из текста структурированное саммари на русском языке. В саммари обязательно выдели следующие пункты (можно использовать маркированные списки):

1. Основные обсуждавшиеся темы или вопросы
2. Ключевые аргументы, предложения или идеи (если были)
3. Принятые решения (если таковые были)
4. Поставленные задачи с указанием ответственных лиц (если это можно однозначно понять)
5. Главные выводы или итоги

Формат вывода - Markdown с заголовками."""

    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        return result.get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса к Ollama: {e}")
        print("Убедитесь, что Ollama запущен: ollama serve")
        sys.exit(1)


def save_summary(summary, output_path):
    """Сохранение саммари в Markdown файл"""
    base = Path(output_path).stem
    
    # Убираем суффикс _tagged если он есть
    if base.endswith("_tagged"):
        base = base[:-7]
    
    md_path = Path(output_path).parent / f"{base}_summary.md"
    
    content = f"""# Саммари: {base}

---

## Основные темы

## Ключевые идеи

## Решения

## Задачи

## Выводы

---

*Сгенерировано автоматически с помощью Ollama + Gemma*

*Исходный файл: {output_path}*
"""
    
    # Добавляет саммари в соответствующие разделы
    content = content.replace("## Основные темы", f"## Основные темы\n\n{summary}")
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Саммари сохранено: {md_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Путь к тегированному JSON файлу")
    p.add_argument("--model", default="gemma3:27b", help="Имя модели Ollama")
    p.add_argument("--max-chars", type=int, default=15000, help="Макс. длина текста для LLM")
    
    args = p.parse_args()
    
    if not os.path.isfile(args.path):
        print(f"Ошибка: файл не найден - {args.path}")
        sys.exit(1)
    
    # Загрузка тегированного транскрипта
    print("Загрузка транскрипции...")
    tagged = load_tagged_json(args.path)
    
    # Форматирование для LLM
    print("Форматирование транскрипции...")
    text = format_transcript(tagged, max_chars=args.max_chars)
    
    print(f"Длина текста: {len(text)} символов")
    
    # Генерация саммари
    print(f"Генерация саммари с помощью {args.model}...")
    summary = generate_summary(text, model=args.model)
    
    # Сохранение
    save_summary(summary, args.path)
    
    # Вывод на экран
    print("\n--- САММАРИ ---")
    print(summary)


if __name__ == "__main__":
    main()