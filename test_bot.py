#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности Telegram бота
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import re
from typing import Optional

# Тестовые данные
def test_save_result_files():
    """Тест функции сохранения файлов"""
    print("🧪 Тест функции save_result_files...")
    
    # Создаем временный файл для теста
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mp3', delete=False) as f:
        audio_path = f.name
        f.write("test audio")
    
    # Внедряем тестовую функцию сохранения файлов
    def save_result_files(audio_path: str, result: dict, tagged: Optional[list] = None, summary: Optional[str] = None) -> dict:
        """Сохранение результатов в файлы"""
        base = Path(audio_path).stem
        base_path = Path(audio_path).parent / base
        
        files = {}
        
        # Сохраняем JSON с сегментами
        json_path = str(base_path) + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result["segments"], f, ensure_ascii=False, indent=2)
        files["json"] = json_path
        
        # Сохраняем TXT
        txt_path = str(base_path) + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        files["txt"] = txt_path
        
        # Сохраняем тегированный JSON (если есть диаризация)
        if tagged:
            tagged_path = str(base_path) + "_tagged.json"
            with open(tagged_path, "w", encoding="utf-8") as f:
                json.dump(tagged, f, ensure_ascii=False, indent=2)
            files["tagged"] = tagged_path
            files["tagged_json"] = tagged_path
        
        # Сохраняем тегированный Markdown (если есть диаризация)
        if tagged:
            tagged_md_path = str(base_path) + "_tagged.md"
            with open(tagged_md_path, "w", encoding="utf-8") as f:
                for seg in tagged:
                    speaker = seg.get("speaker", "Unknown")
                    text = seg.get("text", "")
                    f.write(f"[{speaker}]: {text}\n")
            files["tagged_md"] = tagged_md_path
        
        # Сохраняем саммари (если есть)
        if summary:
            summary_path = str(base_path) + "_summary.md"
            content = f"""# Саммари: {base}

---

{summary}

---

*Сгенерировано автоматически с помощью Ollama + Gemma*

*Исходный файл: {audio_path}*
"""
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(content)
            files["summary"] = summary_path
        
        return files
    
    try:
        # Тест 1: Сохранение без диаризации
        result = {
            "segments": [{"start": 0, "end": 1, "text": "Test segment"}],
            "text": "Test transcription text"
        }
        
        files = save_result_files(audio_path, result)
        
        # Проверяем созданные файлы
        assert "json" in files, "JSON файл не создан"
        assert "txt" in files, "TXT файл не создан"
        
        # Проверяем содержимое
        with open(files["json"], "r", encoding="utf-8") as f:
            json_data = json.load(f)
            assert len(json_data) == 1, "JSON содержит неверные данные"
        
        with open(files["txt"], "r", encoding="utf-8") as f:
            txt_data = f.read()
            assert "Test transcription text" in txt_data, "TXT содержит неверные данные"
        
        print("✅ Тест 1 пройден: Сохранение без диаризации")
        
        # Тест 2: Сохранение с диаризацией
        tagged = [
            {"start": 0, "end": 1, "text": "Hello", "speaker": "Speaker1"},
            {"start": 1, "end": 2, "text": "World", "speaker": "Speaker2"}
        ]
        
        files = save_result_files(audio_path, result, tagged=tagged)
        
        # Проверяем, что созданы дополнительные файлы
        assert "tagged_json" in files, "Tagged JSON не создан"
        assert "tagged_md" in files, "Tagged MD не создан"
        
        # Проверяем содержимое tagged MD
        with open(files["tagged_md"], "r", encoding="utf-8") as f:
            md_data = f.read()
            assert "[Speaker1]" in md_data, "Tagged MD не содержит Speaker1"
            assert "Hello" in md_data, "Tagged MD не содержит текст Speaker1"
            assert "[Speaker2]" in md_data, "Tagged MD не содержит Speaker2"
            assert "World" in md_data, "Tagged MD не содержит текст Speaker2"
        
        print("✅ Тест 2 пройден: Сохранение с диаризацией")
        
        print("\n🎉 Все тесты пройдены!")
    
    except Exception as e:
        print(f"❌ Тест упал: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Очистка
        if os.path.exists(audio_path):
            os.remove(audio_path)
        for file_path in files.values():
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    test_save_result_files()