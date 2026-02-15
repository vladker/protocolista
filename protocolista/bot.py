#!/usr/bin/env python3
"""
Telegram Bot Application Module
"""

from telegram.ext import Application

from protocolista.config import TELEGRAM_BOT_TOKEN, BOT_TIMEOUTS


def create_application() -> Application:
    """
    Создание и настройка приложения Telegram Bot

    Returns:
        Настроенное приложение Telegram Bot
    """
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не установлен")

    # Создаем приложение с настройкой таймаутов
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(BOT_TIMEOUTS["connect"])
        .read_timeout(BOT_TIMEOUTS["read"])
        .write_timeout(BOT_TIMEOUTS["write"])
        .pool_timeout(BOT_TIMEOUTS["pool"])
        .get_updates_connect_timeout(BOT_TIMEOUTS["connect"])
        .get_updates_read_timeout(BOT_TIMEOUTS["read"])
        .get_updates_write_timeout(BOT_TIMEOUTS["write"])
        .get_updates_pool_timeout(BOT_TIMEOUTS["pool"])
        .build()
    )

    return application
