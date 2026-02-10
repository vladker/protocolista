#!/usr/bin/env python3
"""
Конвертация тегированного JSON в Markdown
"""

import argparse
import json
import os
from pathlib import Path


def convert_tagged_json_to_txt_md(json_path):
    """Конвертация тегированного JSON в TXT и MD файлы"""
    
    with open(json_path, encoding="utf-8") as f:
        segments = json.load(f)
    
    # Формирование текста для TXT
    txt_lines = []
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        txt_lines.append(f"[{speaker}] ({start:.2f}-{end:.2f}): {text}")
    
    txt_content = "\n".join(txt_lines)
    
    # Формирование Markdown
    md_lines = []
    md_lines.append("# Транскрипция")
    md_lines.append("")
    md_lines.append("| Время | Спикер | Текст |")
    md_lines.append("|-------|--------|-------|")
    
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        time_range = f"{start:.2f}-{end:.2f}"
        md_lines.append(f"| {time_range} | {speaker} | {text} |")
    
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("*Сгенерировано из JSON с помощью convert.py*")
    
    # Сохранение
    base = os.path.splitext(json_path)[0]
    
    txt_path = base + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"TXT сохранен: {txt_path}")
    
    md_path = base + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown сохранен: {md_path}")


def main():
    p = argparse.ArgumentParser(description="Конвертация тегированного JSON в TXT/MD")
    p.add_argument("path", help="Путь к тегированному JSON файлу")
    
    args = p.parse_args()
    
    if not os.path.isfile(args.path):
        print(f"Файл не найден: {args.path}")
        sys.exit(1)
    
    convert_tagged_json_to_txt_md(args.path)


if __name__ == "__main__":
    main()