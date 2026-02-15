#!/usr/bin/env python3
"""
Main entry point for Protocolista Telegram Bot
"""

import os
import sys
import logging

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from protocolista module
from protocolista import (
    TELEGRAM_BOT_TOKEN,
    BOT_TIMEOUTS,
    LOG_FORMAT,
    LOG_DATEFMT,
    LOG_LEVEL,
)
from protocolista.bot import create_application
from protocolista.logging import setup_logging

# Настройка логирования
setup_logging(LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT)
logger = logging.getLogger(__name__)


def validate_config():
    """Проверка необходимых конфигурационных параметров"""
    errors = []

    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN не установлен")

    return errors


def main():
    """Запуск бота"""
    logger.info("=" * 60)
    logger.info("🤖 Запуск Telegram бота для обработки аудиофайлов")
    logger.info("=" * 60)

    # Валидация конфигурации
    config_errors = validate_config()
    if config_errors:
        for error in config_errors:
            logger.error(f"❌ Ошибка конфигурации: {error}")
        sys.exit(1)

    logger.info("✅ Конфигурация валидна")

    # Создание и запуск приложения
    try:
        application = create_application()

        logger.info("📦 Бот настроен, добавление обработчиков...")
        from protocolista.handlers import register_handlers

        register_handlers(application)

        logger.info("✅ Обработчики добавлены")
        logger.info("⏳ Запуск бота (polling)...")
        logger.info("🤖 Бот запущен и работает. Ожидание сообщений...")
        logger.info("Нажмите Ctrl+C для остановки")

        application.run_polling(allowed_updates=None)

    except KeyboardInterrupt:
        logger.info("👋 Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске бота: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
