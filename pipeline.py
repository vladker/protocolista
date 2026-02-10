#!/usr/bin/env python3
"""
Основной скрипт пайплайна обработки аудио
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_pipeline(
    filepath,
    use_clean=True,
    do_summary=True,
    lang="ru",
    temperature=0,
    beam_size=None,
    condition=False,
    prompt=""
):
    """Запуск полного пайплайна обработки"""
    
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    
    # Детектирование модели в зависимости от GPU памяти
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem < 8:
            whisper_model = "tiny"
        elif gpu_mem < 12:
            whisper_model = "medium"
        else:
            whisper_model = "large-v3"
    else:
        whisper_model = "tiny"
    
    base_name = os.path.splitext(filepath)[0]
    cleaned_file = base_name + "_cleaned.wav"
    audio_file = cleaned_file if use_clean else filepath
    
    print("=" * 50)
    print("ПАЙПЛАЙН ОБРАБОТКИ АУДИО")
    print("=" * 50)
    print(f"Исходный файл: {filepath}")
    print(f"Используемый файл: {audio_file}")
    print()
    
    # Шаг 1: Очистка аудио (опционально)
    if use_clean:
        print("[ШАГ 1/4] Очистка аудио от тишины...")
        subprocess.run(["python3", "clean_audio.py", filepath])
        print()
    
    # Шаг 2: Транскрипция Whisper
    print(f"[ШАГ 2/4] Транскрипция с Whisper (модель: {whisper_model})...")
    cmd = [
        "python3", "transcribe.py",
        audio_file,
        "--lang", lang,
        "--temperature", str(temperature),
        "--model", whisper_model,
    ]
    if beam_size:
        cmd += ["--beam_size", str(beam_size)]
    if condition:
        cmd.append("--condition")
    if prompt:
        cmd += ["--prompt", prompt]
    subprocess.run(cmd)
    print()
    
    # Шаг 3: Диаризация NeMo (опционально)
    print("[ШАГ 3/4] Диаризация спикеров с NeMo...")
    json_file = os.path.splitext(audio_file)[0] + ".json"
    result = subprocess.run(["python3", "diarize.py", audio_file, json_file, "12"])
    if result.returncode != 0:
        print("Диаризация пропущена (NeMo не установлен или ошибка)")
    print()
    
    # Шаг 4: Конвертация в TXT / MD
    tagged_json = os.path.splitext(audio_file)[0] + "_tagged.json"
    if os.path.exists(tagged_json):
        print("[ШАГ 4/4] Конвертация в Markdown...")
        subprocess.run(["python3", "convert.py", tagged_json])
        print()
    else:
        # Если диаризация не удалась, используем обычный json
        json_file = os.path.splitext(audio_file)[0] + ".json"
        if os.path.exists(json_file):
            print("[ШАГ 4/4] Конвертация в Markdown...")
            subprocess.run(["python3", "convert.py", json_file])
            print()
    
    # Шаг 5: Генерация саммари (опционально)
    if do_summary:
        print("[ДОПОЛНИТЕЛЬНО] Генерация саммари с Ollama...")
        if os.path.exists(tagged_json):
            subprocess.run(["python3", "summarize.py", tagged_json])
        elif os.path.exists(json_file):
            subprocess.run(["python3", "summarize.py", json_file])
        else:
            print("Нет данных для саммари")
        print()
    
    print("=" * 50)
    print("ПАЙПЛАЙН ЗАВЕРШЕН!")
    print("=" * 50)
    print(f"\nРезультаты сохранены в папке: {os.path.dirname(filepath)}")
    print("Файлы:")
    print(f"  - {base_name}.txt (транскрипция)")
    print(f"  - {base_name}_tagged.json (диаризованный текст)")
    print(f"  - {base_name}.md (Markdown версия)")
    if do_summary:
        print(f"  - {base_name}_summary.md (саммари)")


def main():
    p = argparse.ArgumentParser(
        description="Пайплайн обработки аудио: транскрипция + диаризация + саммари"
    )
    p.add_argument("path", help="Путь к аудио- или видеофайлу")
    p.add_argument("--no-clean", action="store_true", help="Не очищать аудио от тишины")
    p.add_argument("--no-summary", action="store_true", help="Не генерировать саммари")
    p.add_argument("--lang", default="ru", help="Язык (по умолчанию: ru)")
    p.add_argument("--temperature", type=float, default=0, help="Температура для Whisper")
    p.add_argument("--beam-size", type=int, help="Beam size для Whisper")
    p.add_argument("--condition", action="store_true", help="condition_on_previous_text")
    p.add_argument("--prompt", default="", help="Вводная для модели")
    
    args = p.parse_args()
    
    try:
        run_pipeline(
            args.path,
            use_clean=not args.no_clean,
            do_summary=not args.no_summary,
            lang=args.lang,
            temperature=args.temperature,
            beam_size=args.beam_size,
            condition=args.condition,
            prompt=args.prompt
        )
    except Exception as e:
        print(f"\nОшибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()