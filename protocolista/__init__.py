#!/usr/bin/env python3
# Protocolista Telegram Bot
"""
Protocolista - Telegram Bot for Audio Processing

Features:
- Audio transcription with Whisper
- Speaker diarization with NeMo
- Summary generation with Ollama
"""

__version__ = "1.0.0"
__author__ = "Vladker"

from protocolista.config import (
    TELEGRAM_BOT_TOKEN,
    WHISPER_MODEL,
    DIARIZATION_MAX_SPEAKERS,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    TEMP_DIR,
    MAX_MESSAGE_LENGTH,
    MAX_TRANSCRIPT_CHARS,
    LOG_FORMAT,
    LOG_DATEFMT,
    LOG_LEVEL,
    SUPPORTED_AUDIO_FORMATS,
    DIARIZATION_MODEL_REPO,
    BOT_TIMEOUTS,
)

from protocolista.utils import (
    extract_embeddings,
    auto_cluster,
    merge_segments,
    format_transcript,
    clean_speakers_from_text,
    get_temp_dir,
    generate_temp_filename,
    cleanup_file,
    save_json,
    load_json,
    validate_audio_format,
    validate_file_exists,
    format_duration,
    get_timestamp,
)

__all__ = [
    # Config
    "TELEGRAM_BOT_TOKEN",
    "WHISPER_MODEL",
    "DIARIZATION_MAX_SPEAKERS",
    "OLLAMA_API_URL",
    "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT",
    "TEMP_DIR",
    "MAX_MESSAGE_LENGTH",
    "MAX_TRANSCRIPT_CHARS",
    "LOG_FORMAT",
    "LOG_DATEFMT",
    "LOG_LEVEL",
    "SUPPORTED_AUDIO_FORMATS",
    "DIARIZATION_MODEL_REPO",
    "BOT_TIMEOUTS",
    # Utils
    "extract_embeddings",
    "auto_cluster",
    "merge_segments",
    "format_transcript",
    "clean_speakers_from_text",
    "get_temp_dir",
    "generate_temp_filename",
    "cleanup_file",
    "save_json",
    "load_json",
    "validate_audio_format",
    "validate_file_exists",
    "format_duration",
    "get_timestamp",
]
