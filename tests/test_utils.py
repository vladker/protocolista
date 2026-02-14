"""Тесты для утилит"""

import pytest
import sys
import tempfile
from pathlib import Path
import os
import json
from unittest.mock import patch, MagicMock

# Импортируем функции из основного модуля
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import telegram_bot as bot


class TestUtils:
    """Тесты для утилитарных функций"""

    @pytest.mark.unit
    def test_save_result_files_json(self):
        """Тест сохранения результатов в JSON файл"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test_audio.wav")
            with open(audio_path, "w") as f:
                f.write("fake audio")

            result = {"text": "Текст транскрипции", "segments": [{"text": "Сегмент", "start": 0, "end": 10}]}

            files = bot.save_result_files(audio_path, result)

            assert "json" in files
            assert os.path.exists(files["json"])

            with open(files["json"], "r", encoding="utf-8") as f:
                content = json.load(f)
                assert content == result["segments"]

    @pytest.mark.unit
    def test_save_result_files_txt(self):
        """Тест сохранения результатов в TXT файл"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test_audio.wav")
            with open(audio_path, "w") as f:
                f.write("fake audio")

            result = {"text": "Текст транскрипции", "segments": []}

            files = bot.save_result_files(audio_path, result)

            assert "txt" in files
            assert os.path.exists(files["txt"])

            with open(files["txt"], "r", encoding="utf-8") as f:
                content = f.read()
                assert "Текст транскрипции" in content

    @pytest.mark.unit
    def test_save_result_files_with_tagged(self):
        """Тест сохранения результатов с диаризацией"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test_audio.wav")
            with open(audio_path, "w") as f:
                f.write("fake audio")

            result = {"text": "Текст транскрипции", "segments": []}

            tagged = [
                {"text": "Сегмент 1", "start": 0, "end": 10, "speaker": "Speaker1"},
                {"text": "Сегмент 2", "start": 10, "end": 20, "speaker": "Speaker2"},
            ]

            files = bot.save_result_files(audio_path, result, tagged=tagged)

            assert "tagged" in files
            assert "tagged_md" in files
            assert "tagged_json" in files

            with open(files["tagged_md"], "r", encoding="utf-8") as f:
                content = f.read()
                assert "Speaker1:" in content
                assert "Speaker2:" in content

    @pytest.mark.unit
    def test_save_result_files_with_summary(self):
        """Тест сохранения результатов с саммари"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test_audio.wav")
            with open(audio_path, "w") as f:
                f.write("fake audio")

            result = {"text": "Текст транскрипции", "segments": []}

            summary = "# Саммари\n\nТекст саммари"

            files = bot.save_result_files(audio_path, result, summary=summary)

            assert "summary" in files
            assert os.path.exists(files["summary"])

            with open(files["summary"], "r", encoding="utf-8") as f:
                content = f.read()
                assert "Саммари" in content
                assert "Текст саммари" in content

    @pytest.mark.unit
    def test_auto_cluster_basic(self):
        """Тест авто-кластеризации"""
        # Создаем простые эмбеддинги
        import numpy as np

        embs = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        labels = bot.auto_cluster(embs, max_k=3)

        assert len(labels) == len(embs)
        assert all(isinstance(l, int) for l in labels)

    @pytest.mark.unit
    def test_auto_cluster_max_k(self):
        """Тест авто-кластеризации с max_k"""
        import numpy as np

        embs = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])

        # Проверяем, что не превышается max_k
        labels = bot.auto_cluster(embs, max_k=3)

        assert len(labels) == len(embs)

    @pytest.mark.unit
    def test_merge_segments_basic(self):
        """Тест слияния сегментов"""
        stamps = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
        labels = [0, 0, 1]

        merged = bot.merge_segments(stamps, labels)

        assert len(merged) == 2
        assert merged[0]["spk"] == 0
        assert merged[1]["spk"] == 1

    @pytest.mark.unit
    def test_merge_segments_with_gap(self):
        """Тест слияния сегментов с разрывом"""
        stamps = [(0.0, 5.0), (5.5, 10.0), (10.0, 15.0)]
        labels = [0, 0, 1]

        merged = bot.merge_segments(stamps, labels, gap=0.5)

        assert len(merged) == 3  # Разрыв 0.5 больше gap, сегменты не сливаются

    @pytest.mark.unit
    def test_merge_segments_no_gap(self):
        """Тест слияния сегментов без разрыва"""
        stamps = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
        labels = [0, 0, 1]

        merged = bot.merge_segments(stamps, labels, gap=0.0)

        assert len(merged) == 2  # Сегменты 0 и 1 сливаются

    @pytest.mark.unit
    def test_format_transcript_empty_segments(self):
        """Тест форматирования пустого списка сегментов"""
        result = bot.format_transcript([])

        assert result == ""

    @pytest.mark.unit
    def test_format_transcript_unicode(self):
        """Тест форматирования текста с unicode"""
        tagged_segments = [
            {"speaker": "Speaker1", "text": "Привет мир 🌍"},
            {"speaker": "Speaker2", "text": "Спасибо за помощь ❤️"},
        ]

        result = bot.format_transcript(tagged_segments)

        assert "🌍" in result
        assert "❤️" in result

    @pytest.mark.unit
    def test_format_transcript_empty_text(self):
        """Тест форматирования сегмента с пустым текстом"""
        tagged_segments = [{"speaker": "Speaker1", "text": ""}]

        result = bot.format_transcript(tagged_segments)

        assert "Speaker1:" in result
        assert "Текст без спикера" in result

    @pytest.mark.unit
    def test_get_whisper_model_creates_temp_dir(self):
        """Тест создания временной директории"""
        original_temp = bot.TEMP_DIR

        # Очищаем директорию если существует
        if bot.TEMP_DIR.exists():
            for file in bot.TEMP_DIR.glob("*"):
                file.unlink()

        try:
            with patch("telegram_bot.whisper.load_model", return_value=MagicMock()):
                model = bot.get_whisper_model()
                assert model is not None

            assert bot.TEMP_DIR.exists()
        finally:
            # Восстанавливаем
            bot.TEMP_DIR = original_temp

    @pytest.mark.unit
    def test_generate_summary_timeout(self):
        """Тест генерации саммари с таймаутом"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test_audio.wav")
            with open(audio_path, "w") as f:
                f.write("fake audio")

            result = {"text": "Текст транскрипции", "segments": []}

            tagged = [{"text": "Сегмент", "start": 0, "end": 10, "speaker": "Speaker1"}]

            # Проверяем сохранение файлов
            files = bot.save_result_files(audio_path, result, tagged=tagged)

            # Проверяем, что файлы созданы
            assert "tagged_md" in files
            assert os.path.exists(files["tagged_md"])
