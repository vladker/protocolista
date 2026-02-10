#!/usr/bin/env python3
"""
Транскрипция аудиофайла с помощью Whisper (OpenAI)
"""

import argparse
import json
import os
import time
import torch
import whisper


def main():
    # Аргументы командной строки
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Аудио- или видеофайл")
    p.add_argument("--lang", default="ru", help="Язык (по умолчанию: ru)")
    p.add_argument("--model", default="medium", help="Модель Whisper (medium для 8GB GPU, large-v3 для 16GB+)")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--beam_size", type=int)
    p.add_argument("--condition", action="store_true", help="condition_on_previous_text")
    p.add_argument("--prompt", default="", help="Вводная: тема, участники, термины...")
    
    args = p.parse_args()
    
    if not os.path.isfile(args.path):
        raise SystemExit(f"Файл не найден: {args.path}")
    
    # Выбор устройства (GPU или CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device.upper()} - {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    
    # Загрузка модели
    print(f"Загрузка модели {args.model}...")
    model = whisper.load_model(args.model, device=device)
    
    # Транскрибация
    print("Транскрибирование...")
    t0 = time.time()
    
    result = model.transcribe(
        args.path,
        language=args.lang,
        temperature=args.temperature,
        beam_size=args.beam_size,
        condition_on_previous_text=args.condition,
        initial_prompt=args.prompt or None
    )
    
    elapsed = round(time.time() - t0, 2)
    print(f"Время обработки: {elapsed} сек")
    
    # Сохранение результатов
    base = os.path.splitext(args.path)[0]
    
    # Сохранение в JSON с сегментами
    json_path = base + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    print(f"Сохранено: {json_path}")
    
    # Сохранение в TXT
    txt_path = base + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"Сохранено: {txt_path}")
    
    # Вывод результата
    print("\n--- РЕЗУЛЬТАТ ---")
    print(result["text"])


if __name__ == "__main__":
    main()