#!/usr/bin/env python3
"""
Telegram Bot Handlers Module
"""

import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from protocolista import config


def register_handlers(application: Application) -> None:
    """
    Регистрация обработчиков команд и сообщений

    Args:
        application: Приложение Telegram Bot
    """
    # Команды
    from protocolista.commands import start, help_command

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Обработчик аудиофайлов
    application.add_handler(
        MessageHandler(filters.VOICE | filters.AUDIO | (filters.Document.AUDIO & ~filters.COMMAND), handle_audio)
    )

    # Команды обработки транскрипции
    from protocolista.commands import (
        s2t_command,
        s2t_spk_command,
        md_command,
        list_command,
        summary_command,
        summary_md_command,
        protocol_command,
        protocol_md_command,
    )

    application.add_handler(CommandHandler("s2t", s2t_command))
    application.add_handler(CommandHandler("s2t_spk", s2t_spk_command))
    application.add_handler(CommandHandler("md", md_command))
    application.add_handler(CommandHandler("list", list_command))
    application.add_handler(CommandHandler("summary", summary_command))
    application.add_handler(CommandHandler("summary_md", summary_md_command))
    application.add_handler(CommandHandler("protocol", protocol_command))
    application.add_handler(CommandHandler("protocol_md", protocol_md_command))

    # Обработчик ошибок
    application.add_error_handler(error_handler)


async def handle_audio(update: Update, context) -> None:
    """Обработка аудиофайла"""
    from protocolista.audio import process_audio

    chat_id = update.effective_chat.id

    # Отправляем уведомление о начале обработки
    await context.bot.send_message(chat_id=chat_id, text="📥 Получил файл! Начинаю обработку...")

    try:
        # Скачиваем файл
        file = update.message.voice or update.message.audio or update.message.document
        file_obj = await context.bot.get_file(file.file_id, read_timeout=120)

        # Создаем уникальное имя для файла
        file_extension = os.path.splitext(file.file_name)[1] if file.file_name else ".mp3"
        from protocolista.config import TEMP_DIR

        audio_path = str(TEMP_DIR / f"{chat_id}_{file.file_unique_id}{file_extension}")

        # Скачиваем
        await file_obj.download_to_drive(audio_path)

        # Обрабатываем аудио
        await process_audio(audio_path, lang=config.WHISPER_LANGUAGE)

        # Отправляем результат (заглушка)
        await context.bot.send_message(chat_id=chat_id, text="Аудио обработано!")

        # Удаляем временные файлы
        if os.path.exists(audio_path):
            os.remove(audio_path)

    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка при обработке файла: {e}")


async def error_handler(update: object, context) -> None:
    """Логирование ошибок с информацией о Telegram API"""
    import logging

    logger = logging.getLogger(__name__)

    logger.error(f"Ошибка обработчика: {context.error}")
    if update and isinstance(update, Update):
        logger.error(f"Update: {update}")
