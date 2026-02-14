"""Тесты для команд бота"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock
from telegram import Update
from telegram.ext import ContextTypes

# Импортируем функции из основного модуля
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import telegram_bot as bot


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

        await bot.start(update, context)

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

        await bot.help_command(update, context)

        # Проверяем, что сообщение было отправлено
        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Помощь" in call_args[1]["text"]
        assert "/start" in call_args[1]["text"]
        assert "/help" in call_args[1]["text"]

    @pytest.mark.unit
    async def test_s2t_command_no_user_data(self):
        """Тест команды /s2t без данных пользователя"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        await bot.s2t_command(update, context)

        # Проверяем, что сообщение об ошибке было отправлено
        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Сначала отправьте аудиофайл" in call_args[1]["text"]

    @pytest.mark.unit
    async def test_s2t_command_no_txt_file(self):
        """Тест команды /s2t без файла транскрипции"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        # Создаем данные пользователя без файла txt
        bot.user_data_store[123] = {"files": {}}

        await bot.s2t_command(update, context)

        # Проверяем, что сообщение об ошибке было отправлено
        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Расшифровка еще не готова" in call_args[1]["text"]

        # Очищаем данные пользователя
        bot.user_data_store.clear()

    @pytest.mark.unit
    async def test_s2t_command_success(self):
        """Тест успешной работы команды /s2t"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        # Создаем данные пользователя с файлом txt
        test_text = "Тестовая транскрипция\n\n[Speaker1]: Привет\n[Speaker2]: Здравствуйте"
        bot.user_data_store[123] = {"files": {"txt": "/tmp/test.txt", "tagged_json": "/tmp/test_tagged.json"}}

        # Создаем временный файл
        with open("/tmp/test.txt", "w", encoding="utf-8") as f:
            f.write(test_text)

        try:
            await bot.s2t_command(update, context)

            # Проверяем, что сообщение было отправлено
            context.bot.send_message.assert_called()
        finally:
            # Очищаем данные пользователя и временные файлы
            bot.user_data_store.clear()
            for path in ["/tmp/test.txt", "/tmp/test_tagged.json"]:
                if os.path.exists(path):
                    os.unlink(path)

    @pytest.mark.unit
    async def test_s2t_spk_command_no_user_data(self):
        """Тест команды /s2t_spk без данных пользователя"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        await bot.s2t_spk_command(update, context)

        # Проверяем, что сообщение об ошибке было отправлено
        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Сначала отправьте аудиофайл" in call_args[1]["text"]

    @pytest.mark.unit
    async def test_md_command_no_user_data(self):
        """Тест команды /md без данных пользователя"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        await bot.md_command(update, context)

        # Проверяем, что сообщение об ошибке было отправлено
        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Сначала отправьте аудиофайл" in call_args[1]["text"]

    @pytest.mark.unit
    async def test_list_command_no_user_data(self):
        """Тест команды /list без данных пользователя"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        await bot.list_command(update, context)

        # Проверяем, что сообщение об ошибке было отправлено
        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Сначала отправьте аудиофайл" in call_args[1]["text"]

    @pytest.mark.unit
    async def test_summary_command_no_user_data(self):
        """Тест команды /summary без данных пользователя"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        await bot.summary_command(update, context)

        # Проверяем, что сообщение об ошибке было отправлено
        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Сначала отправьте аудиофайл" in call_args[1]["text"]

    @pytest.mark.unit
    async def test_protocol_command_no_user_data(self):
        """Тест команды /protocol без данных пользователя"""
        update = MagicMock(spec=Update)
        update.effective_chat.id = 123

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot.send_message = AsyncMock()

        await bot.protocol_command(update, context)

        # Проверяем, что сообщение об ошибке было отправлено
        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 123
        assert "Сначала отправьте аудиофайл" in call_args[1]["text"]

    @pytest.mark.unit
    def test_format_markdown_basic(self):
        """Тест форматирования Markdown"""
        text = "Hello *world* and [link](url)"
        result = bot.format_markdown(text)

        assert r"\*world\*" in result
        assert r"\[link\](url)" in result

    @pytest.mark.unit
    def test_format_markdown_special_chars(self):
        """Тест форматирования Markdown с спецсимволами"""
        text = "Text with `code`, #header, **bold**"
        result = bot.format_markdown(text)

        assert r"\`code\`" in result
        assert r"#header" in result
        assert r"\*\*bold\*\*" in result
