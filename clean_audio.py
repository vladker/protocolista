#!/usr/bin/env python3
"""
Очистка аудио от тишины с помощью ffmpeg
"""

import argparse
import os
import subprocess
import sys


def clean_audio(input_path, output_path=None):
    """Очистка аудио от длительных пауз (тишины)"""
    
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + "_cleaned.wav"
    
    # Команда ffmpeg для преобразования в моно 16kHz и удаления пауз
    cmd = [
        "ffmpeg",
        "-i", str(input_path),      # входной файл
        "-ac", "1",                 # преобразовать в моно (1 канал)
        "-ar", "16000",             # частота дискретизации 16 кГц
        "-af",                      # аудио фильтр
        "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-35dB:detection=peak",
        str(output_path)
    ]
    
    print(f"Очистка аудио: {input_path}")
    print(f"Выход: {output_path}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Ошибка: {result.stderr}")
            return False
        
        print(f"Очищенное аудио сохранено: {output_path}")
        return True
        
    except FileNotFoundError:
        print("Ошибка: ffmpeg не найден. Установите ffmpeg.")
        return False


def main():
    p = argparse.ArgumentParser(description="Очистка аудио от тишины")
    p.add_argument("input", help="Входной аудиофайл")
    p.add_argument("-o", "--output", help="Выходной файл (по умолчанию: input_cleaned.wav)")
    
    args = p.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Файл не найден: {args.input}")
        sys.exit(1)
    
    success = clean_audio(args.input, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()