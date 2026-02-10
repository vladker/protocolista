# Локальный транскрибатор аудио с саммари

Этот проект представляет собой пайплайн для автоматической обработки аудиофайлов: транскрипция, диаризация спикеров и генерация саммари.

## Компоненты

- **Whisper** — распознавание речи (OpenAI)
- **NVIDIA NeMo** — диаризация спикеров (кто что говорил)
- **Ollama + Gemma 27B** — генерация саммари
- **Obsidian** — хранение результатов

## Установка

### Предварительные требования

- Python 3.9+
- FFmpeg (для очистки аудио)
- CUDA-совместимая видеокарта (рекомендуется для ускорения)
- 16+ ГБ оперативной памяти
- 50+ ГБ свободного места на диске для моделей

### Шаг 1: Установка FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:** Скачайте с [ffmpeg.org](https://ffmpeg.org/download.html)

### Шаг 2: Установка Python-зависимостей

```bash
pip install -r requirements.txt
```

### Шаг 3: Установка Ollama

Скачайте и установите Ollama с официального сайта: [https://ollama.com](https://ollama.com)

Запустите Ollama и загрузите модель Gemma 27B:
```bash
ollama pull gemma3:27b
```

### Шаг 4: Установка Obsidian (опционально)

Скачайте и установите Obsidian: [https://obsidian.md](https://obsidian.md)

## Использование

### Быстрый старт

```bash
python pipeline.py путь/к/аудиофайлу.mp3
```

### Полный запуск с опциями

```bash
python pipeline.py input.wav \
  --lang ru \
  --temperature 0 \
  --no-summary
```

### Опции командной строки

| Параметр | Описание |
|----------|----------|
| `--no-clean` | Не очищать аудио от тишины |
| `--no-summary` | Не генерировать саммари |
| `--lang` | Язык (по умолчанию: ru) |
| `--temperature` | Температура для Whisper (0-1) |
| `--beam-size` | Beam size для улучшения качества |
| `--prompt` | Вводная подсказка для модели |

### Отдельные шаги

```bash
# Транскрипция
python transcribe.py audio.mp3 --lang ru

# Диаризация
python diarize.py audio.wav audio.json 12

# Саммари
python summarize.py audio_tagged.json
```

## Структура проекта

```
whisper/
├── transcribe.py      # Транскрипция аудио с помощью Whisper
├── diarize.py         # Диаризация спикеров с помощью NeMo
├── summarize.py       # Генерация саммари с помощью Ollama
├── convert.py         # Конвертация JSON в Markdown
├── pipeline.py        # Основной скрипт пайплайна
├── clean_audio.py     # Очистка аудио от тишины
├── requirements.txt   # Python зависимости
└── README.md
```

## Примеры вывода

### Транскрипция (TXT)
```
Привет, команда! Какие у нас сегодня новости по проекту?
Всё идёт по плану, но есть пара нюансов...
```

### Диаризованный JSON
```json
{
  "speaker": "Speaker1",
  "text": "Привет, команда!",
  "start": 0.0,
  "end": 1.5
}
```

### Саммари (Markdown)
```markdown
# Саммари: meeting

## Основные темы
- Обсуждение прогресса проекта

## Решения
- Продолжить текущий график

## Задачи
- Иван: подготовить отчёт (до пятницы)
```

## Известные проблемы

- **NeMo требует значительных ресурсов** - для больших файлов используйте GPU
- **Ollama должна быть запущена** перед генерацией саммари
- **Модели загружаются один раз** - первый запуск может быть медленным

## Полезные ссылки

- [Whisper GitHub](https://github.com/openai/whisper)
- [NVIDIA NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Ollama](https://ollama.com)
- [Gemma](https://ai.google.dev/gemma)
- [Obsidian](https://obsidian.md)