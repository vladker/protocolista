#!/usr/bin/env python3
"""
Диаризация спикеров с помощью NVIDIA NeMo

ОПЦИОНАЛЬНЫЙ СКРИПТ - диаризация требует много ресурсов
Для простой транскрипции используйте только transcribe.py
"""

import sys
import os

try:
    import signal
    import numpy as np
    import librosa
    import torch
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
except ImportError as e:
    print(f"Ошибка: NeMo не установлен или есть проблемы с зависимостями: {e}")
    print("Установите: pip install nemo_toolkit")
    print("Для простой транскрипции используйте только transcribe.py")
    print("Диаризация пропущена.")
    sys.exit(1)


def extract_embeddings(wav: np.ndarray, sr: int, model: EncDecSpeakerLabelModel,
                       win_s: float = 3.0, step_s: float = 1.5):
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


def auto_cluster(embs: np.ndarray, max_k: int = 10):
    """Авто-кластеризация спикеров"""
    best_lbl, best_sc = None, -1
    
    for k in range(2, max_k + 1):
        lbl = SpectralClustering(n_clusters=k, affinity="nearest_neighbors").fit_predict(embs)
        sc = silhouette_score(embs, lbl)
        if sc > best_sc:
            best_lbl, best_sc = lbl, sc
    
    return best_lbl


def merge_segments(stamps, labels, gap: float = 0.5):
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


def main():
    if len(sys.argv) != 4:
        print("Использование: python diarize.py <audio.wav> <whisper.json> <max_speakers>")
        sys.exit(1)
    
    wav_path, whisper_json, max_k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    
    if not os.path.isfile(wav_path) or not os.path.isfile(whisper_json):
        print("Ошибка: файл не найден.")
        sys.exit(1)
    
    # Выбор устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device.upper()}")
    
    # Загрузка модели NeMo
    print("Загрузка модели NeMo (titanet_large)...")
    repo = "nvidia/speakerverification_en_titanet_large"
    token = os.getenv("HF_TOKEN")
    
    try:
        if token:
            model = EncDecSpeakerLabelModel.from_pretrained(repo, token=token)
        else:
            model = EncDecSpeakerLabelModel.from_pretrained(repo)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        sys.exit(1)
    
    model = model.to(device).eval()
    
    # Чтение аудио
    print("Чтение аудио...")
    wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    
    # Извлечение эмбеддингов
    print("Извлечение эмбеддингов...")
    embs, stamps = extract_embeddings(wav, sr, model)
    
    # Кластеризация
    print(f"Авто-кластеризация 2..{max_k} спикеров...")
    labels = auto_cluster(embs, max_k=max_k)
    spk_cnt = len(set(labels))
    print(f"Выбрано спикеров: {spk_cnt}")
    
    diar = merge_segments(stamps, labels)
    
    # Слияние с транскрипцией Whisper
    print("Слияние с транскрипцией Whisper...")
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
    
    # Сохранение результата
    out_path = os.path.splitext(whisper_json)[0] + "_tagged.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tagged, f, ensure_ascii=False, indent=2)
    
    print(f"Результат сохранен: {out_path}")


if __name__ == "__main__":
    main()