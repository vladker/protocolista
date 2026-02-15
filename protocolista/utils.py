#!/usr/bin/env python3
"""
Утилитарные функции для Project Protocolista
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import numpy as np
import soundfile as sf
import torch

# ==============================================
# Audio Processing Utilities
# ==============================================


def extract_embeddings(wav: np.ndarray, sr: int, model, win_s: float = 3.0, step_s: float = 1.5) -> tuple:
    """
    Извлечение эмбеддингов из аудио

    Args:
        wav: Аудио в формате numpy array
        sr: Частота дискретизации
        model: Модель NeMo для извлечения эмбеддингов
        win_s: Размер окна (секунды)
        step_s: Шаг сдвига окна (секунды)

    Returns:
        Tuple of (embeddings, timestamps)
    """
    embs, stamps = [], []
    t = 0.0
    total_dur = len(wav) / sr

    while t + win_s <= total_dur:
        segment = wav[int(t * sr) : int((t + win_s) * sr)]

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


def auto_cluster(embs: np.ndarray, max_k: int = 10) -> np.ndarray:
    """
    Авто-кластеризация спикеров с использованием Spectral Clustering

    Args:
        embs: Эмбеддинги спикеров
        max_k: Максимальное количество спикеров

    Returns:
        Массив меток кластеров
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    best_lbl, best_sc = None, -1

    for k in range(2, max_k + 1):
        lbl = SpectralClustering(n_clusters=k, affinity="nearest_neighbors").fit_predict(embs)
        sc = silhouette_score(embs, lbl)
        if sc > best_sc:
            best_lbl, best_sc = lbl, sc

    return best_lbl


def merge_segments(stamps: List[tuple], labels: np.ndarray, gap: float = 0.5) -> List[dict]:
    """
    Слияние последовательных сегментов одного спикера

    Args:
        stamps: Список кортежей (start, end) для каждого сегмента
        labels: Метки спикеров для каждого сегмента
        gap: Максимальный разрыв для слияния (секунды)

    Returns:
        Список словарей сmerged сегментами
    """
    merged = []
    if not stamps or not labels:
        return merged

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


# ==============================================
# Transcript Formatting Utilities
# ==============================================


def format_transcript(tagged_segments: List[dict], max_chars: int = 15000) -> str:
    """
    Форматирование транскрипции для генерации саммари

    Args:
        tagged_segments: Список сегментов с указанием спикера
        max_chars: Максимальная длина результата

    Returns:
        Отформатированный текст
    """
    result = []
    for seg in tagged_segments:
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        result.append(f"[{speaker}]: {text}")

    full_text = "\n".join(result)

    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n... (сокращено)"

    return full_text


def clean_speakers_from_text(text: str) -> str:
    """
    Удаление указаний спикеров из текста

    Args:
        text: Исходный текст

    Returns:
        Очищенный текст
    """
    import re

    # Убираем [Speaker1], [Speaker2] и т.д.
    text = re.sub(r"\[Speaker\d+\]\s*", "", text)
    text = re.sub(r"\(.*?\):\s*", "", text)
    return text.strip()


# ==============================================
# File Utilities
# ==============================================


def get_temp_dir() -> Path:
    """Получение директории для временных файлов"""
    temp_dir = Path(tempfile.gettempdir()) / "telegram_bot_audio"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def generate_temp_filename(suffix: str = ".wav") -> str:
    """Генерация уникального имени для временного файла"""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        return tmp.name


def cleanup_file(filepath: str) -> bool:
    """
    Удаление файла

    Args:
        filepath: Путь к файлу

    Returns:
        True если файл удален, False иначе
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    except Exception:
        return False


# ==============================================
# JSON Utilities
# ==============================================


def save_json(data: dict, filepath: str, ensure_ascii: bool = True, indent: int = 2) -> None:
    """
    Сохранение данных в JSON файл

    Args:
        data: Данные для сохранения
        filepath: Путь к файлу
        ensure_ascii: Экранировать ли非-ASCII символы
        indent: Отступ для форматирования
    """
    with open(filepath, "w", encoding="utf-8" if ensure_ascii else "utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def load_json(filepath: str) -> dict:
    """
    Загрузка данных из JSON файла

    Args:
        filepath: Путь к файлу

    Returns:
        Загруженные данные
    """
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ==============================================
# Validation Utilities
# ==============================================


def validate_audio_format(filename: str, supported_formats: tuple = (".mp3", ".wav", ".m4a", ".ogg")) -> bool:
    """
    Проверка формата аудиофайла

    Args:
        filename: Имя файла
        supported_formats: Поддерживаемые форматы

    Returns:
        True если формат поддерживается
    """
    return Path(filename).suffix.lower() in supported_formats


def validate_file_exists(filepath: str) -> bool:
    """
    Проверка существования файла

    Args:
        filepath: Путь к файлу

    Returns:
        True если файл существует
    """
    return os.path.isfile(filepath)


# ==============================================
# Time Utilities
# ==============================================


def format_duration(seconds: float) -> str:
    """
    Форматирование продолжительности в читаемый формат

    Args:
        seconds: Количество секунд

    Returns:
        Строка формата "MM:SS"
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def get_timestamp() -> str:
    """
    Получение текущей метки времени

    Returns:
        Строка с текущим временем
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
