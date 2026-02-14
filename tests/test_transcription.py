"""Тесты для модуля транскрибации аудио"""

import pytest
import sys
import tempfile
from pathlib import Path
import os
from unittest.mock import patch, MagicMock

# Импортируем функции из основного модуля
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import telegram_bot as bot


class TestTranscription:
    """Тесты для функций транскрибации"""

    @pytest.mark.unit
    def test_format_transcript_basic(self):
        """Тест форматирования базовой транскрипции"""
        tagged_segments = [
            {"speaker": "Speaker1", "text": "Привет"},
            {"speaker": "Speaker2", "text": "Здравствуйте"},
        ]

        result = bot.format_transcript(tagged_segments)

        assert "Speaker1: Привет" in result
        assert "Speaker2: Здравствуйте" in result

    @pytest.mark.unit
    def test_format_transcript_max_chars(self):
        """Тест форматирования с ограничением по длине"""
        # Создаем длинный текст
        long_text = " ".join([f"Текст {i}" for i in range(1000)])
        tagged_segments = [{"speaker": "Speaker1", "text": long_text}]

        result = bot.format_transcript(tagged_segments, max_chars=100)

        assert len(result) <= 100
        assert "..." in result

    @pytest.mark.unit
    def test_format_transcript_unknown_speaker(self):
        """Тест форматирования с неизвестным спикером"""
        tagged_segments = [{"speaker": "Unknown", "text": "Текст без спикера"}]

        result = bot.format_transcript(tagged_segments)

        assert "[Unknown]: Текст без спикера" in result

    @pytest.mark.unit
    def test_transcribe_audio_with_whisper(self):
        """Тест транскрибации с моком Whisper"""
        # Создаем временный аудиофайл
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio data")
            audio_path = tmp.name

        try:
            # Мокаем функции
            with patch.object(bot, "whisper", MagicMock()):
                with patch.object(bot, "get_whisper_model", return_value=MagicMock()):
                    with patch("telegram_bot.whisper.load_model") as mock_load:
                        mock_model = MagicMock()
                        mock_model.transcribe.return_value = {"text": "Транскрибация", "segments": []}
                        mock_load.return_value = mock_model

                    result = bot.transcribe_audio(audio_path)

                    assert "text" in result
                    assert "segments" in result
                    assert mock_model.transcribe.called
        finally:
            os.unlink(audio_path)

    @pytest.mark.unit
    def test_transcribe_audio_without_whisper(self):
        """Тест транскрибации без установленного Whisper"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio data")
            audio_path = tmp.name

        try:
            # Сохраняем текущий импорт
            original_whisper = bot.whisper

            # Устанавливаем None
            bot.whisper = None

            with pytest.raises(ImportError):
                bot.transcribe_audio(audio_path)

            # Восстанавливаем импорт
            bot.whisper = original_whisper
        finally:
            os.unlink(audio_path)

    @pytest.mark.unit
    def test_transcribe_audio_with_custom_language(self):
        """Тест транскрибации с кастомным языком"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio data")
            audio_path = tmp.name

        try:
            with patch.object(bot, "whisper", MagicMock()):
                with patch.object(bot, "get_whisper_model", return_value=MagicMock()):
                    with patch("telegram_bot.whisper.load_model") as mock_load:
                        mock_model = MagicMock()
                        mock_model.transcribe.return_value = {"text": "Транскрибация", "segments": []}
                        mock_load.return_value = mock_model

                    result = bot.transcribe_audio(audio_path, lang="en")

                    assert mock_model.transcribe.called
                    call_kwargs = mock_model.transcribe.call_args[1]
                    assert call_kwargs.get("language") == "en"
        finally:
            os.unlink(audio_path)

    @pytest.mark.unit
    def test_get_whisper_model_cpu(self):
        """Тест получения модели Whisper на CPU"""
        with patch("telegram_bot.torch", None):
            with patch("telegram_bot.whisper.load_model", return_value=MagicMock()):
                model = bot.get_whisper_model()
                assert model is not None

    @pytest.mark.unit
    def test_get_whisper_model_cuda(self):
        """Тест получения модели Whisper на CUDA"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = MagicMock()

        with patch("telegram_bot.torch", mock_torch):
            with patch("telegram_bot.whisper.load_model", return_value=MagicMock()):
                model = bot.get_whisper_model()
                assert model is not None

    @pytest.mark.unit
    def test_get_whisper_model_cuda_out_of_memory(self):
        """Тест получения модели Whisper при нехватке памяти CUDA"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = MagicMock()

        # Симулируем ошибку out of memory
        with patch("telegram_bot.whisper.load_model") as mock_load:
            mock_load.side_effect = RuntimeError("CUDA out of memory")

            # Попытка загрузить меньшую модель
            with patch.object(bot, "WHISPER_MODEL", "medium"):
                with patch("telegram_bot.whisper.load_model") as mock_load_small:
                    mock_load_small.return_value = MagicMock()
                    model = bot.get_whisper_model()
                    assert model is not None

    @pytest.mark.unit
    def test_format_transcript_special_chars(self):
        """Тест форматирования текста со специальными символами"""
        tagged_segments = [
            {"speaker": "Speaker1", "text": "Текст с спецсимволами: @user #hashtag"},
            {"speaker": "Speaker2", "text": "Много-много текста"},
        ]

        result = bot.format_transcript(tagged_segments)

        assert "Speaker1: Текст с спецсимволами: @user #hashtag" in result
        assert "Speaker2: Много-много текста" in result
