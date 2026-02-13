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

# Словарь для хранения обработанных данных пользователей
user_data_store = {}

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
        
        # Сохраняем данные пользователя для последующих команд
        user_data_store[chat_id] = {
            "files": files,
            "tagged_json": files.get("tagged") if "tagged" in files else None,
            "result": result,
            "tagged": tagged
        }
        
        # Финальное сообщение
        await context.bot.send_message(
            chat_id=chat_id,
            text="✅ Обработка завершена! Результаты отправлены выше.\n\nИспользуйте команды:\n- `/s2t` - расшифровка без спикеров\n- `/s2t_spk` - расшифровка со спикерами\n- `/md` - расшифровка в Markdown\n- `/list` - расшифровка со списками\n- `/summary` - сводный протокол в чат\n- `/summary_md` - сводный протокол в файл\n- `/protocol` - протокол встречи в чат\n- `/protocol_md` - протокол встречи в файл"
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
    
    # Добавляем команды обработки транскрипции
    application.add_handler(CommandHandler("s2t", s2t_command))  # Расшифровка без спикеров
    application.add_handler(CommandHandler("s2t_spk", s2t_spk_command))  # Расшифровка со спикерами
    
    # Добавляем команды форматирования в MD
    application.add_handler(CommandHandler("md", md_command))  # Расшифровка в MD
    application.add_handler(CommandHandler("list", list_command))  # Расшифровка со списками
    
    # Добавляем команды протоколов
    application.add_handler(CommandHandler("summary", summary_command))  # Сводный протокол в чат
    application.add_handler(CommandHandler("summary_md", summary_md_command))  # Сводный протокол в MD
    application.add_handler(CommandHandler("protocol", protocol_command))  # Протокол в чат
    application.add_handler(CommandHandler("protocol_md", protocol_md_command))  # Протокол в MD
    
    logger.info("✅ Обработчики добавлены")
    logger.info("⏳ Запуск бота (polling)...")
    logger.info("🤖 Бот запущен и работает. Ожидание сообщений...")
    logger.info("Нажмите Ctrl+C для остановки")
    
    # Запускаем бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)


async def s2t_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /s2t - Расшифровать запись без указания говорящих
    Получает расшифровку для последнего обработанного файла пользователя
    """
    chat_id = update.effective_chat.id
    
    if chat_id not in user_data_store:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Сначала отправьте аудиофайл для обработки."
        )
        return
    
    user_data = user_data_store[chat_id]
    if "txt" not in user_data["files"]:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Расшифровка еще не готова. Отправьте аудиофайл для обработки."
        )
        return
    
    try:
        with open(user_data["files"]["txt"], "r", encoding="utf-8") as f:
            text = f.read()
        
        # Очищаем от указаний спикеров
        clean_lines = []
        for line in text.split("\n"):
            # Убираем [Speaker1], [Speaker2] и т.д.
            import re
            line = re.sub(r'\[Speaker\d+\]\s*', '', line)
            line = re.sub(r'\(.*?\):\s*', '', line)
            clean_lines.append(line.strip())
        
        clean_text = "\n".join(clean_lines)
        
        # Отправляем расшифровку без спикеров
        max_len = 4000
        if len(clean_text) > max_len:
            for i in range(0, len(clean_text), max_len):
                await context.bot.send_message(chat_id=chat_id, text=clean_text[i:i + max_len])
        else:
            await context.bot.send_message(chat_id=chat_id, text=f"```txt\n{clean_text}\n```", parse_mode="MarkdownV2")
        
        logger.info(f"Команда /s2t обработана для chat_id: {chat_id}")
    
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при обработке команды /s2t: {e}"
        )
        logger.error(f"Ошибка /s2t: {e}")


async def s2t_spk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /s2t_spk - Расшифровать запись с указанием говорящих
    Получает тегированную расшифровку для последнего обработанного файла пользователя
    """
    chat_id = update.effective_chat.id
    
    if chat_id not in user_data_store:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Сначала отправьте аудиофайл для обработки."
        )
        return
    
    user_data = user_data_store[chat_id]
    if "tagged" not in user_data["files"]:
        # Пытаемся получить тегированную расшифровку
        if not user_data.get("tagged_json"):
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Расшифровка со спикерами еще не готова. Отправьте аудиофайл для обработки."
            )
            return
    else:
        if "txt" not in user_data["files"]:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Расшифровка еще не готова. Отправьте аудиофайл для обработки."
            )
            return
    
    try:
        # Проверяем, есть ли уже обработанная тегированная расшифровка
        if "tagged_md" in user_data["files"]:
            with open(user_data["files"]["tagged_md"], "r", encoding="utf-8") as f:
                text = f.read()
        else:
            with open(user_data["files"]["txt"], "r", encoding="utf-8") as f:
                text = f.read()
        
        # Отправляем расшифровку со спикерами
        max_len = 4000
        if len(text) > max_len:
            for i in range(0, len(text), max_len):
                await context.bot.send_message(chat_id=chat_id, text=text[i:i + max_len])
        else:
            await context.bot.send_message(chat_id=chat_id, text=f"```txt\n{text}\n```", parse_mode="MarkdownV2")
        
        logger.info(f"Команда /s2t_spk обработана для chat_id: {chat_id}")
    
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при обработке команды /s2t_spk: {e}"
        )
        logger.error(f"Ошибка /s2t_spk: {e}")


async def md_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /md - Отправить расшифровку в формате Markdown
    Получает MD файл для последнего обработанного файла пользователя
    """
    chat_id = update.effective_chat.id
    
    if chat_id not in user_data_store:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Сначала отправьте аудиофайл для обработки."
        )
        return
    
    user_data = user_data_store[chat_id]
    if "md" not in user_data["files"]:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ MD файл еще не готов. Отправьте аудиофайл для обработки."
        )
        return
    
    try:
        with open(user_data["files"]["md"], "r", encoding="utf-8") as f:
            text = f.read()
        
        # Отправляем Markdown
        max_len = 4000
        if len(text) > max_len:
            for i in range(0, len(text), max_len):
                await context.bot.send_message(chat_id=chat_id, text=f"```md\n{text[i:i + max_len]}\n```", parse_mode="MarkdownV2")
        else:
            await context.bot.send_message(chat_id=chat_id, text=f"```md\n{text}\n```", parse_mode="MarkdownV2")
        
        logger.info(f"Команда /md обработана для chat_id: {chat_id}")
    
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при обработке команды /md: {e}"
        )
        logger.error(f"Ошибка /md: {e}")


async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /list - Отправить расшифровку со списками в Markdown
    Создает расшифровку с маркированными списками для последнего обработанного файла пользователя
    """
    chat_id = update.effective_chat.id
    
    if chat_id not in user_data_store:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Сначала отправьте аудиофайл для обработки."
        )
        return
    
    user_data = user_data_store[chat_id]
    if "txt" not in user_data["files"]:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Расшифровка еще не готова. Отправьте аудиофайл для обработки."
        )
        return
    
    try:
        with open(user_data["files"]["txt"], "r", encoding="utf-8") as f:
            text = f.read()
        
        # Форматируем с маркированными списками
        list_lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                # Добавляем маркер списка, если строка не пустая
                import re
                # Убираем указания спикеров для чистоты
                clean_line = re.sub(r'\[Speaker\d+\]\s*', '', line)
                clean_line = re.sub(r'\(.*?\):\s*', '', clean_line)
                if clean_line:
                    list_lines.append(f"- {clean_line}")
        
        list_text = "\n".join(list_lines)
        
        # Отправляем Markdown со списками
        max_len = 4000
        if len(list_text) > max_len:
            for i in range(0, len(list_text), max_len):
                await context.bot.send_message(chat_id=chat_id, text=f"```md\n{list_text[i:i + max_len]}\n```", parse_mode="MarkdownV2")
        else:
            await context.bot.send_message(chat_id=chat_id, text=f"```md\n{list_text}\n```", parse_mode="MarkdownV2")
        
        logger.info(f"Команда /list обработана для chat_id: {chat_id}")
    
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при обработке команды /list: {e}"
        )
        logger.error(f"Ошибка /list: {e}")


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /summary - Выдать сводный протокол в чат
    Получает саммари для последнего обработанного файла пользователя
    """
    chat_id = update.effective_chat.id
    
    if chat_id not in user_data_store:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Сначала отправьте аудиофайл для обработки."
        )
        return
    
    user_data = user_data_store[chat_id]
    if "summary" not in user_data["files"]:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Сводный протокол еще не готов. Отправьте аудиофайл для обработки."
        )
        return
    
    try:
        with open(user_data["files"]["summary"], "r", encoding="utf-8") as f:
            text = f.read()
        
        # Отправляем сводный протокол
        max_len = 4000
        if len(text) > max_len:
            for i in range(0, len(text), max_len):
                await context.bot.send_message(chat_id=chat_id, text=f"```md\n{text[i:i + max_len]}\n```", parse_mode="MarkdownV2")
        else:
            await context.bot.send_message(chat_id=chat_id, text=f"```md\n{text}\n```", parse_mode="MarkdownV2")
        
        logger.info(f"Команда /summary обработана для chat_id: {chat_id}")
    
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при обработке команды /summary: {e}"
        )
        logger.error(f"Ошибка /summary: {e}")


async def summary_md_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /summary_md - Выдать сводный протокол в формате Markdown
    Отправляет файл саммари как документ
    """
    chat_id = update.effective_chat.id
    
    if chat_id not in user_data_store:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Сначала отправьте аудиофайл для обработки."
        )
        return
    
    user_data = user_data_store[chat_id]
    if "summary" not in user_data["files"]:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Сводный протокол еще не готов. Отправьте аудиофайл для обработки."
        )
        return
    
    try:
        summary_path = user_data["files"]["summary"]
        logger.info(f"Отправка файла саммари: {summary_path}")
        await context.bot.send_document(
            chat_id=chat_id,
            document=open(summary_path, "rb"),
            filename="summary.md"
        )
        logger.info(f"Команда /summary_md обработана для chat_id: {chat_id}")
    
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при обработке команды /summary_md: {e}"
        )
        logger.error(f"Ошибка /summary_md: {e}")


async def protocol_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /protocol - Выдать протокол в чат на основе шаблона встречи
    Генерирует протокол с использованием шаблона
    """
    chat_id = update.effective_chat.id
    
    if chat_id not in user_data_store:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Сначала отправьте аудиофайл для обработки."
        )
        return
    
    user_data = user_data_store[chat_id]
    
    try:
        # Если есть тегированная расшифровка, используем её
        if "tagged_json" in user_data:
            tagged_path = user_data["tagged_json"]
            with open(tagged_path, "r", encoding="utf-8") as f:
                tagged = json.load(f)
            
            # Форматируем для генерации протокола
            text_for_protocol = format_transcript(tagged)
            
            # Генерируем протокол через Ollama
            protocol_text = generate_protocol(text_for_protocol)
            
            if protocol_text:
                max_len = 4000
                if len(protocol_text) > max_len:
                    for i in range(0, len(protocol_text), max_len):
                        await context.bot.send_message(chat_id=chat_id, text=f"```md\n{protocol_text[i:i + max_len]}\n```", parse_mode="MarkdownV2")
                else:
                    await context.bot.send_message(chat_id=chat_id, text=f"```md\n{protocol_text}\n```", parse_mode="MarkdownV2")
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="❌ Не удалось сгенерировать протокол."
                )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Нет данных для генерации протокола. Отправьте аудиофайл для обработки."
            )
    
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при обработке команды /protocol: {e}"
        )
        logger.error(f"Ошибка /protocol: {e}")


async def protocol_md_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /protocol_md - Выдать протокол в формате Markdown на основе шаблона встречи
    Генерирует и отправляет файл протокола
    """
    chat_id = update.effective_chat.id
    
    if chat_id not in user_data_store:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Сначала отправьте аудиофайл для обработки."
        )
        return
    
    user_data = user_data_store[chat_id]
    
    try:
        # Если есть тегированная расшифровка, используем её
        if "tagged_json" in user_data:
            tagged_path = user_data["tagged_json"]
            with open(tagged_path, "r", encoding="utf-8") as f:
                tagged = json.load(f)
            
            # Форматируем для генерации протокола
            text_for_protocol = format_transcript(tagged)
            
            # Генерируем протокол через Ollama
            protocol_text = generate_protocol(text_for_protocol)
            
            if protocol_text:
                # Сохраняем протокол во временный файл
                protocol_file = str(TEMP_DIR / f"protocol_{chat_id}.md")
                with open(protocol_file, "w", encoding="utf-8") as f:
                    f.write(protocol_text)
                
                # Отправляем файл
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=open(protocol_file, "rb"),
                    filename="protocol.md"
                )
                
                # Удаляем временный файл
                os.remove(protocol_file)
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="❌ Не удалось сгенерировать протокол."
                )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Нет данных для генерации протокола. Отправьте аудиофайл для обработки."
            )
    
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при обработке команды /protocol_md: {e}"
        )
        logger.error(f"Ошибка /protocol_md: {e}")


def generate_protocol(text: str, model: str = "gemma3:27b", timeout: int = 120) -> Optional[str]:
    """Генерация протокола через Ollama API с шаблоном встречи"""
    import requests
    
    logger.info(f"Начинается генерация протокола (модель: {model})")
    logger.info(f"Длина текста для протокола: {len(text)} символов")
    
    system_prompt = """Ты - эксперт по оформлению протоколов встреч и совещаний.

На основе транскрипции встречи создай официальный протокол на русском языке в формате Markdown.

Формат протокола:
# Протокол встречи

## Дата и время
[Указать дату и время встречи, если известны]

## Участники
[Указать участников встречи, если известны]

## Повестка дня
- [Перечислить пункты повестки]

## Обсуждение
[Описывать обсуждение по пунктам повестки]

## Решения
[Список принятых решений]

## Задачи
[Задачи с указанием ответственных и сроков]

## Заключение
[Итоги встречи]
"""

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
                "content": f"Транскрипция встречи:\n\n{text}"
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=(10, timeout))
        response.raise_for_status()
        result = response.json()
        logger.info("Протокол успешно сгенерирован")
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


if __name__ == "__main__":
    main()
