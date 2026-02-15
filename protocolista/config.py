#!/usr/bin/env python3
"""
Конфигурация проекта Protocolista
Централизованное управление настройками
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()


# ==============================================
# Telegram Bot Configuration
# ==============================================

TELEGRAM_BOT_TOKEN: Optional[str] = os.environ.get("TELEGRAM_BOT_TOKEN")

# Таймауты для бота (в секундах)
BOT_TIMEOUTS = {
    "connect": int(os.environ.get("BOT_CONNECT_TIMEOUT", "120")),
    "read": int(os.environ.get("BOT_READ_TIMEOUT", "300")),
    "write": int(os.environ.get("BOT_WRITE_TIMEOUT", "120")),
    "pool": int(os.environ.get("BOT_POOL_TIMEOUT", "120")),
}

# ==============================================
# Whisper Configuration
# ==============================================

WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")
WHISPER_LANGUAGE: str = os.environ.get("WHISPER_LANGUAGE", "ru")

# ==============================================
# NeMo (Diarization) Configuration
# ==============================================

DIARIZATION_MAX_SPEAKERS: int = int(os.environ.get("DIARIZATION_MAX_SPEAKERS", "12"))

# ==============================================
# Ollama Configuration
# ==============================================

OLLAMA_API_URL: str = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma3:27b")
OLLAMA_TIMEOUT: int = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

# ==============================================
# Temporary Files Configuration
# ==============================================

TEMP_DIR: Path = Path(os.environ.get("TEMP_DIR", "/tmp")) / "telegram_bot_audio"
TEMP_DIR.mkdir(exist_ok=True)

# ==============================================
# Telegram Message Limits
# ==============================================

MAX_MESSAGE_LENGTH: int = 4000  # Telegram API limit
MAX_TRANSCRIPT_CHARS: int = 15000  # Для генерации саммари

# ==============================================
# Logging Configuration
# ==============================================

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATEFMT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

# ==============================================
# Supported Audio Formats
# ==============================================

SUPPORTED_AUDIO_FORMATS: tuple = (".mp3", ".wav", ".m4a", ".ogg")

# ==============================================
# Diarization Model Configuration
# ==============================================

DIARIZATION_MODEL_REPO: str = "nvidia/speakerverification_en_titanet_large"
