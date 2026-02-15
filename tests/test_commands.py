"""Тесты для команд бота"""

import pytest
import sys
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update
from telegram.ext import ContextTypes

# Импортируем функции из модуля
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocolista import commands, utils


class TestCommands:
    """Тесты для команд бота"""

    @pytest.mark.unit
    async def test_start_command(self):
        """Тест команды /start"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123
        update.effective_user.id = 456
        update.effective_user.username = "testuser"

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        await commands.start(update, context)

        # Проверяем, что сообщение было отправлено
        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Привет" in call_args[1]["text"]
        assert "Я бот для обработки аудиофайлов" in call_args[1]["text"]

    @pytest.mark.unit
    async def test_help_command(self):
        """Тест команды /help"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123
        update.effective_user.id = 456
        update.effective_user.username = "testuser"

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        await commands.help_command(update, context)

        # Проверяем, что сообщение было отправлено
        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Помощь" in call_args[1]["text"]
        assert "/start" in call_args[1]["text"]
        assert "/help" in call_args[1]["text"]

    @pytest.mark.unit
    async def test_help_command_success(self):
        """Тест успешной работы команды /help"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        await commands.help_command(update, context)

        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "/start" in call_args[1]["text"]
        assert "/help" in call_args[1]["text"]

    @pytest.mark.unit
    async def test_start_command_success(self):
        """Тест успешной работы команды /start"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        await commands.start(update, context)

        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Привет" in call_args[1]["text"]
        assert "Whisper" in call_args[1]["text"]

    @pytest.mark.unit
    def test_format_transcript_basic(self):
        """Тест форматирования транскрипции"""
        tagged_segments = [
            {"speaker": "Speaker1", "text": "Привет"},
            {"speaker": "Speaker2", "text": "Здравствуйте"},
        ]

        result = utils.format_transcript(tagged_segments)

        assert "Speaker1: Привет" in result
        assert "Speaker2: Здравствуйте" in result

    @pytest.mark.unit
    def test_format_transcript_max_chars(self):
        """Тест форматирования с ограничением по длине"""
        long_text = " ".join([f"Текст {i}" for i in range(1000)])
        tagged_segments = [{"speaker": "Speaker1", "text": long_text}]

        result = utils.format_transcript(tagged_segments, max_chars=100)

        assert len(result) <= 100
        assert "..." in result

    @pytest.mark.unit
    def test_format_transcript_empty_segments(self):
        """Тест форматирования пустого списка сегментов"""
        result = utils.format_transcript([])

        assert result == ""

    @pytest.mark.unit
    def test_clean_speakers_from_text(self):
        """Тест удаления указаний спикеров"""
        text = "[Speaker1]: Привет\n[Speaker2]: Здравствуйте"
        result = utils.clean_speakers_from_text(text)

        assert "Speaker1" not in result
        assert "Speaker2" not in result
