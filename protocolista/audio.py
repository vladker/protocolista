#!/usr/bin/env python3
"""
Audio Processing Module
"""

import os
import json
import asyncio
from typing import Optional

import torch

try:
    import whisper
except ImportError:
    whisper = None

try:
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
except ImportError:
    EncDecSpeakerLabelModel = None

from protocolista import config, utils


def get_whisper_model():
    """Получить модель Whisper с определением устройства"""
    device = "cuda" if whisper and torch and torch.cuda.is_available() else "cpu"

    try:
        return whisper.load_model(config.WHISPER_MODEL, device=device)
    except RuntimeError as e:
        if "CUDA" in str(e) and "out of memory" in str(e).lower():
            # Попытка загрузить меньшую модель
            for model_size in ["small", "base"]:
                try:
                    return whisper.load_model(model_size, device=device)
                except RuntimeError:
                    continue
            return whisper.load_model("base", device="cpu")
        else:
            raise


def transcribe_audio(audio_path: str, lang: str = "ru") -> dict:
    """Транскрипция аудио с помощью Whisper"""
    if whisper is None:
        raise ImportError("Whisper не установлен")

    model = get_whisper_model()
    result = model.transcribe(audio_path, language=lang)
    return result


async def process_audio_async(audio_path: str, lang: str = "ru") -> dict:
    """Асинхронная транскрипция аудио с помощью Whisper"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, transcribe_audio, audio_path, lang)


async def diarize_audio_async(audio_path: str, whisper_json: str, max_speakers: int = 12) -> Optional[dict]:
    """Асинхронная диаризация спикеров с помощью NeMo"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, diarize_audio, audio_path, whisper_json, max_speakers)


def diarize_audio(audio_path: str, whisper_json: str, max_speakers: int = 12) -> Optional[dict]:
    """Диаризация спикеров с помощью NeMo"""
    if EncDecSpeakerLabelModel is None:
        return None

    if torch is None:
        return None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = config.DIARIZATION_MODEL_REPO

    model = EncDecSpeakerLabelModel.from_pretrained(repo)
    model = model.to(device).eval()

    # Извлечение эмбеддингов
    import librosa

    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    embs, stamps = utils.extract_embeddings(wav, sr, model)

    # Кластеризация
    labels = utils.auto_cluster(embs, max_k=max_speakers)

    diar = utils.merge_segments(stamps, labels)

    # Слияние с транскрипцией Whisper
    with open(whisper_json, encoding="utf-8") as f:
        whisper_segs = json.load(f)

    tagged = []
    for seg in whisper_segs:
        spk = next(
            (f"Speaker{d['spk'] + 1}" for d in diar if not (seg["end"] <= d["s"] or seg["start"] >= d["e"])),
            "Unknown",
        )
        tagged.append({**seg, "speaker": spk})

    return tagged


async def process_audio(audio_path: str, lang: str = "ru") -> dict:
    """Полная обработка аудио (транскрипция + диаризация + саммари)"""
    result = await process_audio_async(audio_path, lang)

    whisper_json = audio_path.replace(os.path.splitext(audio_path)[1], ".json")
    with open(whisper_json, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)

    tagged = None
    if EncDecSpeakerLabelModel is not None:
        tagged = await diarize_audio_async(audio_path, whisper_json, config.DIARIZATION_MAX_SPEAKERS)

    return {"result": result, "tagged": tagged}
