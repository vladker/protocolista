#!/usr/bin/env python3
"""
Telegram бот для обработки аудиофайлов
- Транскрипция речи (Whisper)
- Диаризация спикеров (NeMo)
- Генерация саммари (Ollama + Gemma)
"""

import os
import sys
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()
import tempfile
import asyncio
import traceback
from pathlib import Path
from typing import Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from telegram import Update, MessageEntity
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# Импортируем функции из существующих модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:
    torch = None

try:
    import whisper
except ImportError:
    whisper = None

try:
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    import numpy as np
    import librosa
except ImportError:
    EncDecSpeakerLabelModel = None


# Пути к моделям
# По умолчанию используем base модель для меньшего потребления памяти
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
DIARIZATION_MAX_SPEAKERS = int(os.environ.get("DIARIZATION_MAX_SPEAKERS", "12"))

# Временная директория для обработки файлов
TEMP_DIR = Path(tempfile.gettempdir()) / "telegram_bot_audio"
TEMP_DIR.mkdir(exist_ok=True)


def get_whisper_model():
    """Получить модель Whisper с определением устройства"""
    global WHISPER_MODEL
    
    logger.info(f"Загрузка модели Whisper: {WHISPER_MODEL}")
    if torch is None:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Если используется CUDA, очищаем кэш перед загрузкой модели
    if device == "cuda" and torch.cuda.is_available():
        logger.info("Очистка кэша CUDA перед загрузкой модели...")
        torch.cuda.empty_cache()
    
    logger.info(f"Используемое устройство: {device}")
    
    try:
        return whisper.load_model(WHISPER_MODEL, device=device)
    except RuntimeError as e:
        if "CUDA" in str(e) and "out of memory" in str(e).lower():
            logger.warning(f"Не хватает памяти GPU для модели {WHISPER_MODEL}, пробуем уменьшить размер модели...")
            # Попытка загрузить меньшую модель
            # Проверяем текущую модель в списке и начинаем с меньшей
            if WHISPER_MODEL == "medium":
                small_models = ["small", "base"]
            elif WHISPER_MODEL == "small":
                small_models = ["base"]
            else:
                small_models = ["base"]  # base - наименьшая модель
            
            for model_size in small_models:
                logger.info(f"Попытка загрузить модель {model_size}...")
                try:
                    WHISPER_MODEL = model_size
                    model = whisper.load_model(model_size, device=device)
                    logger.info(f"Модель {model_size} загружена успешно")
                    return model
                except RuntimeError as inner_e:
                    if "CUDA" not in str(inner_e) or "out of memory" not in str(inner_e).lower():
                        logger.info(f"Модель {model_size} загружена успешно")
                        WHISPER_MODEL = model_size
                        return whisper.load_model(model_size, device=device)
                    logger.warning(f"Не удалось загрузить модель {model_size}: {inner_e}")
            
            # Если все модели CUDA не работают, пробуем CPU
            logger.warning("Все модели CUDA не работают, переключаемся на CPU...")
            WHISPER_MODEL = WHISPER_MODEL  # Оставляем текущую модель
            return whisper.load_model(WHISPER_MODEL, device="cpu")
        else:
            raise


def transcribe_audio(audio_path: str, lang: str = "ru") -> dict:
    """Транскрипция аудио с помощью Whisper"""
    if whisper is None:
        raise ImportError("Whisper не установлен")
    
    logger.info(f"Начинается транскрипция аудио: {audio_path}")
    model = get_whisper_model()
    result = model.transcribe(audio_path, language=lang)
    logger.info("Транскрипция завершена")
    return result


def diarize_audio(audio_path: str, whisper_json: str, max_speakers: int = 12) -> Optional[dict]:
    """Диаризация спикеров с помощью NeMo"""
    if EncDecSpeakerLabelModel is None:
        logger.info("NeMo не установлен, пропускаем диаризацию")
        return None
    
    if torch is None:
        logger.info("PyTorch не установлен, пропускаем диаризацию")
        return None
    
    # Очистка кэша CUDA перед диаризацией
    if torch.cuda.is_available():
        logger.info("Очистка кэша CUDA перед диаризацией...")
        torch.cuda.empty_cache()
    
    logger.info("Начинается диаризация спикеров...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используемое устройство для диаризации: {device}")
        repo = "nvidia/speakerverification_en_titanet_large"
        
        model = EncDecSpeakerLabelModel.from_pretrained(repo)
        model = model.to(device).eval()
        
        # Извлечение эмбеддингов
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        embs, stamps = extract_embeddings(wav, sr, model)
        
        # Кластеризация
        labels = auto_cluster(embs, max_k=max_speakers)
        spk_cnt = len(set(labels))
        
        diar = merge_segments(stamps, labels)
        logger.info(f"Обнаружено спикеров: {spk_cnt}")
        
        # Слияние с транскрипцией Whisper
        with open(whisper_json, encoding="utf-8") as f:
            whisper_segs = json.load(f)
        
        tagged = []
        for seg in whisper_segs:
            spk = next(
                (f"Speaker{d['spk'] + 1}" for d in diar
                 if not (seg['end'] <= d['s'] or seg['start'] >= d['e'])),
                "Unknown"
            )
            tagged.append({**seg, "speaker": spk})
        
        logger.info("Диаризация завершена")
        return tagged
    
    except Exception as e:
        logger.error(f"Ошибка диаризации: {e}")
        return None


def extract_embeddings(wav, sr, model, win_s=3.0, step_s=1.5):
    """Извлечение эмбеддингов из аудио"""
    import soundfile as sf
    import tempfile
    
    embs, stamps = [], []
    t = 0.0
    total_dur = len(wav) / sr
    
    while t + win_s <= total_dur:
        segment = wav[int(t * sr): int((t + win_s) * sr)]
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, segment, sr)
            tmp_path = tmp.name
        
        try:
            with torch.no_grad():
                emb = model.get_embedding(tmp_path).cpu().numpy().squeeze()
            embs.append(emb / np.linalg.norm(emb))
            stamps.append((t, t + win_s))
        finally:
            os.remove(tmp_path)
        
        t += step_s
    
    return np.stack(embs), stamps


def auto_cluster(embs, max_k=10):
    """Авто-кластеризация спикеров"""
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    
    best_lbl, best_sc = None, -1
    
    for k in range(2, max_k + 1):
        lbl = SpectralClustering(n_clusters=k, affinity="nearest_neighbors").fit_predict(embs)
        sc = silhouette_score(embs, lbl)
        if sc > best_sc:
            best_lbl, best_sc = lbl, sc
    
    return best_lbl


def merge_segments(stamps, labels, gap=0.5):
    """Слияние последовательных сегментов одного спикера"""
    merged = []
    cur = {"spk": int(labels[0]), "s": stamps[0][0], "e": stamps[0][1]}
    
    for (s, e), lab in zip(stamps[1:], labels[1:]):
        lab = int(lab)
        if lab == cur["spk"] and s <= cur["e"] + gap:
            cur["e"] = e
        else:
            merged.append(cur)
            cur = {"spk": lab, "s": s, "e": e}
    
    merged.append(cur)
    return merged


def format_transcript(tagged_segments: list, max_chars: int = 15000) -> str:
    """Форматирование транскрипции для саммари"""
    result = []
    for seg in tagged_segments:
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        result.append(f"[{speaker}]: {text}")
    
    full_text = "\n".join(result)
    
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n... (сокращено)"
    
    return full_text


def generate_summary(text: str, model: str = "gemma3:27b", timeout: int = 120) -> Optional[str]:
    """Генерация саммари через Ollama API"""
    import requests
    
    logger.info(f"Начинается генерация саммари (модель: {model})")
    logger.info(f"Длина текста для саммари: {len(text)} символов")
    
    system_prompt = """Ты мой эффективный AI-ассистент по анализу стенограмм совещаний и лекций.

Сделай из текста структурированное саммари на русском языке. В саммари обязательно выдели следующие пункты (можно использовать маркированные списки):

1. Основные обсуждавшиеся темы или вопросы
2. Ключевые аргументы, предложения или идеи (если были)
3. Принятые решения (если таковые были)
4. Поставленные задачи с указанием ответственных лиц (если это можно однозначно понять)
5. Главные выводы или итоги

Формат вывода - Markdown с заголовками."""

    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=(10, timeout))  # (connect_timeout, read_timeout)
        response.raise_for_status()
        result = response.json()
        logger.info("Саммари успешно сгенерировано")
        return result.get("message", {}).get("content", "")
    except requests.exceptions.Timeout:
        logger.error(f"Ollama API timeout after {timeout} seconds")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Ollama API недоступен. Убедитесь, что Ollama запущен на http://localhost:11434")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка Ollama API: {e}")
        return None


def save_result_files(
    audio_path: str,
    result: dict,
    tagged: Optional[list] = None,
    summary: Optional[str] = None
) -> dict:
    """Сохранение результатов в файлы"""
    logger.info("Сохранение результатов в файлы...")
    base = Path(audio_path).stem
    base_path = Path(audio_path).parent / base
    
    files = {}
    
    # Сохраняем JSON с сегментами
    json_path = str(base_path) + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    files["json"] = json_path
    logger.info(f"Сохранен JSON: {json_path}")
    
    # Сохраняем TXT
    txt_path = str(base_path) + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    files["txt"] = txt_path
    logger.info(f"Сохранен TXT: {txt_path}")
    
    # Сохраняем тегированный JSON (если есть диаризация)
    if tagged:
        tagged_path = str(base_path) + "_tagged.json"
        with open(tagged_path, "w", encoding="utf-8") as f:
            json.dump(tagged, f, ensure_ascii=False, indent=2)
        files["tagged"] = tagged_path
        logger.info(f"Сохранен тегированный JSON: {tagged_path}")
    
    # Сохраняем саммари (если есть)
    if summary:
        summary_path = str(base_path) + "_summary.md"
        content = f"""# Саммари: {base}

---

{summary}

---

*Сгенерировано автоматически с помощью Ollama + Gemma*

*Исходный файл: {audio_path}*
"""
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(content)
        files["summary"] = summary_path
        logger.info(f"Сохранено саммари: {summary_path}")
    
    logger.info(f"Сохранено файлов: {len(files)}")
    return files


async def send_result(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    result: dict,
    files: dict,
    summary: Optional[str] = None,
    tagged: Optional[list] = None
):
    """Отправка результатов пользователю"""
    chat_id = update.effective_chat.id
    logger.info(f"Отправка результатов пользователю (chat_id: {chat_id})")
    
    # Отправляем транскрипцию (TXT)
    if "txt" in files:
        with open(files["txt"], "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Транскрипция отправлена (длина: {len(text)} символов)")
        
        # Telegram имеет ограничение на длину сообщения (4096 символов)
        # Разбиваем на части
        max_len = 4000
        if len(text) > max_len:
            for i in range(0, len(text), max_len):
                await context.bot.send_message(chat_id=chat_id, text=text[i:i + max_len])
        else:
            await context.bot.send_message(chat_id=chat_id, text=f"```txt\n{text}\n```", parse_mode="MarkdownV2")
    
    # Отправляем саммари
    if summary:
        logger.info("Саммари отправлено")
        await context.bot.send_message(chat_id=chat_id, text=f"```md\n# Саммари:\n\n{summary}\n```", parse_mode="MarkdownV2")
    
    # Отправляем файлы
    if "json" in files:
        logger.info("Файл JSON отправлен")
        await context.bot.send_document(chat_id=chat_id, document=open(files["json"], "rb"), filename="transcription.json")
    
    if "tagged" in files:
        logger.info("Файл диаризации (tagged) отправлен")
        await context.bot.send_document(chat_id=chat_id, document=open(files["tagged"], "rb"), filename="diarized.json")
    
    if "summary" in files:
        logger.info("Файл саммари отправлен")
        await context.bot.send_document(chat_id=chat_id, document=open(files["summary"], "rb"), filename="summary.md")
    
    logger.info("Все результаты отправлены пользователю")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    chat_id = update.effective_chat.id
    user = update.effective_user
    
    logger.info(f"Команда /start от пользователя (id: {user.id}, username: {user.username}, chat_id: {chat_id})")
    
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
    await context.bot.send_message(
        chat_id=chat_id,
        text=welcome_message,
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help"""
    chat_id = update.effective_chat.id
    user = update.effective_user
    
    logger.info(f"Команда /help от пользователя (id: {user.id}, username: {user.username}, chat_id: {chat_id})")
    
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
    await context.bot.send_message(
        chat_id=chat_id,
        text=help_message,
        parse_mode="Markdown"
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка аудиофайла"""
    chat_id = update.effective_chat.id
    
    logger.info(f"Получен новый аудиофайл от пользователя (chat_id: {chat_id})")
    
    # Отправляем уведомление о начале обработки
    sent_message = await context.bot.send_message(
        chat_id=chat_id,
        text="📥 Получил файл! Начинаю обработку..."
    )
    
    try:
        # Скачиваем файл
        file = update.message.voice or update.message.audio or update.message.document
        
        # Используем timeout для загрузки файла (на случай медленного соединения)
        file_obj = await context.bot.get_file(file.file_id, read_timeout=120)
        
        # Создаем уникальное имя для файла
        file_extension = os.path.splitext(file.file_name)[1] if file.file_name else ".mp3"
        audio_path = str(TEMP_DIR / f"{chat_id}_{file.file_unique_id}{file_extension}")
        
        logger.info(f"Скачивание аудио: {file.file_name} ({file.file_size} байт)")
        # Скачиваем
        await file_obj.download_to_drive(audio_path)
        logger.info(f"Аудио скачано: {audio_path}")
        
        # Обновляем статус
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=sent_message.message_id,
            text="🎤 Транскрибирую аудио (Whisper)... Это может занять несколько минут."
        )
        
        # Транскрипция
        result = transcribe_audio(audio_path, lang="ru")
        
        # Обновляем статус
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=sent_message.message_id,
            text="👥 Разделяю речь по спикерам (NeMo)... (опционально)"
        )
        
        # Диаризация (опционально)
        whisper_json = audio_path.replace(file_extension, ".json")
        tagged = None
        if EncDecSpeakerLabelModel is not None:
            tagged = diarize_audio(audio_path, whisper_json, DIARIZATION_MAX_SPEAKERS)
        
        # Обновляем статус
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=sent_message.message_id,
            text="📝 Генерирую саммари (Ollama)... (опционально)"
        )
        
        # Генерация саммари (опционально)
        summary = None
        if tagged:
            text_for_summary = format_transcript(tagged)
            summary = generate_summary(text_for_summary)
        
        # Сохраняем файлы
        files = save_result_files(audio_path, result, tagged, summary)
        
        # Удаляем исходный файл
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Отправляем результат
        await send_result(update, context, result, files, summary, tagged)
        
        # Удаляем временные файлы
        for file_type, file_path in files.items():
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Финальное сообщение
        await context.bot.send_message(
            chat_id=chat_id,
            text="✅ Обработка завершена! Результаты отправлены выше."
        )
    
    except Exception as e:
        error_msg = str(e)[:500]  # Ограничиваем сообщение об ошибке
        error_trace = traceback.format_exc()[:1000]  # Ограничиваем стек трейс
        error_text = f"❌ Ошибка при обработке файла:\n\n{error_msg}\n\n{error_trace}"
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=sent_message.message_id,
            text=error_text
        )
        logger.error(f"Ошибка при обработке файла: {e}")
        logger.error(f"Стек трейса: {error_trace}")


def main():
    """Запуск бота"""
    logger.info("=" * 60)
    logger.info("🤖 Запуск Telegram бота для обработки аудиофайлов")
    logger.info("=" * 60)
    
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    
    if not token:
        logger.error("❌ Ошибка: TELEGRAM_BOT_TOKEN не установлен")
        logger.error("Пожалуйста, установите переменную окружения TELEGRAM_BOT_TOKEN")
        logger.error("Пример: export TELEGRAM_BOT_TOKEN='123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11'")
        return
    
    logger.info("✅ Токен бота получен")
    
    # Создаем приложение с настройкой таймаута
    # Увеличенные таймауты для обработки долгих операций (транскрипция Whisper, диаризация NeMo)
    application = (
        Application.builder()
        .token(token)
        .connect_timeout(600)   # 60 секунд на подключение
        .read_timeout(3000)     # 5 минут на чтение (для долгих операций)
        .build()
    )
    
    logger.info("📦 Бот настроен, добавление обработчиков...")
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    
    # Обработчик аудиофайлов
    # Слушаем голосовые сообщения, аудио и документы с аудио
    application.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO | (filters.Document.AUDIO & ~filters.COMMAND),
        handle_audio
    ))
    
    logger.info("✅ Обработчики добавлены")
    logger.info("⏳ Запуск бота (polling)...")
    logger.info("🤖 Бот запущен и работает. Ожидание сообщений...")
    logger.info("Нажмите Ctrl+C для остановки")
    
    # Запускаем бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()