#!/usr/bin/env python3
"""
Telegram Bot Command Handlers
"""

from telegram import Update
from telegram.ext import ContextTypes


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /start"""
    chat_id = update.effective_chat.id

    welcome_message = """
Привет! Я бот для обработки аудиофайлов.

Я могу:
- 🎤 Транскрибировать речь из аудио (Whisper)
- 👥 Разделять речь по спикерам (NeMo)
- 📝 Генерировать саммари (Ollama + Gemma)

**Инструкция:**
1. Отправьте мне аудиофайл (mp3, wav, m4a, ogg)
2. Подождите обработки
3. Получите результат: транскрипцию, диаризацию и саммари

**Примечание:**
- Для диаризации и саммари необходимы дополнительные ресурсы (Ollama должна быть запущена)
- Обработка может занять несколько минут в зависимости от длины аудио

*Используйте /help для получения дополнительной информации.*
"""
    await context.bot.send_message(chat_id=chat_id, text=welcome_message, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /help"""
    chat_id = update.effective_chat.id

    help_message = """
**Помощь**

Доступные команды:
- `/start` - Начало работы с ботом
- `/help` - Показать эту справку

Поддерживаемые форматы:
- MP3
- WAV
- M4A
- OGG

**Опции:**
Отправьте аудиофайл, и бот автоматически обработает его.

*Бот использует Whisper для транскрипции, NeMo для диаризации и Ollama для генерации саммари.*
"""
    await context.bot.send_message(chat_id=chat_id, text=help_message, parse_mode="Markdown")


async def s2t_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда /s2t - Расшифровать запись без указания говорящих
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="❌ Команда /s2t требует реализации обработки данных пользователя"
    )


async def s2t_spk_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда /s2t_spk - Расшифровать запись с указанием говорящих
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="❌ Команда /s2t_spk требует реализации обработки данных пользователя"
    )


async def md_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда /md - Отправить расшифровку в формате Markdown
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="❌ Команда /md требует реализации обработки данных пользователя"
    )


async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда /list - Отправить расшифровку со списками в Markdown
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="❌ Команда /list требует реализации обработки данных пользователя"
    )


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда /summary - Выдать сводный протокол в чат
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="❌ Команда /summary требует реализации обработки данных пользователя"
    )


async def summary_md_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда /summary_md - Выдать сводный протокол в формате Markdown
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="❌ Команда /summary_md требует реализации обработки данных пользователя"
    )


async def protocol_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда /protocol - Выдать протокол в чат на основе шаблона встречи
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="❌ Команда /protocol требует реализации обработки данных пользователя"
    )


async def protocol_md_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда /protocol_md - Выдать протокол в формате Markdown на основе шаблона встречи
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="❌ Команда /protocol_md требует реализации обработки данных пользователя",
    )
