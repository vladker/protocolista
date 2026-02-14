# Руководство по тестированию и QA

Этот документ содержит инструкции по использованию инструментов для обеспечения качества кода проекта.

## Содержание

- [Установка зависимостей](#установка-зависимостей)
- [Запуск тестов](#запуск-тестов)
- [Форматирование кода](#форматирование-кода)
- [Линтинг](#линтинг)
- [Проверка типов](#проверка-типов)
- [Покрытие кода](#покрытие-кода)
- [Автоматическая проверка](#автоматическая-проверка)

## Установка зависимостей

Установите основные зависимости:

```bash
pip install -r requirements.txt
```

Установите зависимости для разработки и тестирования:

```bash
pip install -r requirements-dev.txt
```

## Запуск тестов

### Запуск всех тестов

```bash
pytest tests/ -v
```

### Запуск тестов с покрытием

```bash
pytest tests/ --cov=telegram_bot --cov-report=html --cov-report=term-missing
```

### Запуск тестов с определенными фильтрами

```bash
# Только unit тесты
pytest tests/ -v -m unit

# С определенными файлами
pytest tests/test_commands.py -v

# С подробным выводом
pytest tests/ -v --tb=short
```

### Дополнительные опции

```bash
# Показать подробности пропущенных тестов
pytest tests/ -v --tb=short --verbose

# Запустить несколько процессов
pytest tests/ -v -n auto

# Показать статистику
pytest tests/ --cov=telegram_bot --cov-report=term-missing --cov-report=html
```

## Форматирование кода

### Black (автоматическое форматирование)

```bash
# Форматировать все Python файлы
black telegram_bot.py tests/

# Проверить форматирование без изменений
black --check telegram_bot.py tests/

# Проверить и показать разницу
black --diff telegram_bot.py tests/
```

### isort (импорты)

```bash
# Сортировка импортов
isort telegram_bot.py tests/

# Проверить импорты
isort --check-only telegram_bot.py tests/
```

## Линтинг

### Flake8

```bash
# Базовая проверка
flake8 telegram_bot.py tests/

# С подсчетом ошибок
flake8 telegram_bot.py tests/ --count

# С подсветкой синтаксических ошибок (E9, F63, F7, F82)
flake8 telegram_bot.py tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

# С показом строк и номеров
flake8 telegram_bot.py tests/ --show-source --statistics

# С подсчетом сложности (C90)
flake8 telegram_bot.py tests/ --statistics
```

### Yamllint (для YAML файлов)

```bash
# Проверить все YAML файлы
yamllint .

# С конкретным файлом
yamllint .yamllint
```

### Bandit (проверка безопасности)

```bash
# Базовая проверка
bandit -r telegram_bot.py

# С отчетом HTML
bandit -r telegram_bot.py -f html -o report.html

# С определенными уровнями
bandit -r telegram_bot.py -ll
```

## Проверка типов

### Mypy

```bash
# Базовая проверка типов
mypy telegram_bot.py tests/ --config-file .mypy.ini

# С более строгими настройками
mypy telegram_bot.py tests/ --config-file .mypy.ini --strict

# С игнорированием ошибок
mypy telegram_bot.py tests/ --ignore-missing-imports
```

## Покрытие кода

### Генерация отчетов

```bash
# HTML отчет
pytest tests/ --cov=telegram_bot --cov-report=html

# XML отчет (для CI/CD)
pytest tests/ --cov=telegram_bot --cov-report=xml

# Текстовый отчет
pytest tests/ --cov=telegram_bot --cov-report=term-missing

# Отчет в консоли с процентами
pytest tests/ --cov=telegram_bot --cov-report=term --cov-report=html
```

### Откройте HTML отчет

Откройте файл в браузере:

```bash
open htmlcov/index.html
```

## Автоматическая проверка

### Pre-commit hooks

Установите pre-commit hooks:

```bash
pre-commit install
```

Запустите все проверки вручную:

```bash
pre-commit run --all-files
```

### Makefile

Используйте удобные команды из Makefile:

```bash
# Показать доступные команды
make help

# Запуск всех тестов
make test

# Запуск тестов с покрытием
make test-cov

# Форматирование кода
make format

# Линтинг
make lint

# Проверка типов
make type-check

# Генерация отчетов
make report
```

## Полный цикл проверки

```bash
# 1. Форматирование кода
make format

# 2. Запуск тестов
make test

# 3. Проверка покрытия
make test-cov

# 4. Линтинг
make lint

# 5. Проверка типов
make type-check

# 6. Генерация отчетов
make report
```

## Интеграция в CI/CD

### GitHub Actions

Создайте файл `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.13'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests with coverage
      run: pytest tests/ --cov=telegram_bot --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v2
```

### GitLab CI

Создайте файл `.gitlab-ci.yml`:

```yaml
test:
  image: python:3.13
  script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
    - pytest tests/ --cov=telegram_bot --cov-report=xml --cov-report=html
  artifacts:
    paths:
      - htmlcov/
      - coverage.xml
```

## Статистика покрытия

Текущее покрытие кода:

- **telegram_bot.py**: 33%
- **test_commands.py**: 100%
- **test_transcription.py**: 93%
- **test_utils.py**: 96%

**Общее покрытие**: 40%

Цель: увеличить покрытие до 80%+

## Частые проблемы

### Проблема: Пустые строки с пробелами (W293)

**Решение**: Запустите black для автоматического исправления

```bash
black telegram_bot.py tests/
```

### Проблема: Неопределенное имя (F821)

**Решение**: Проверьте импорты и убедитесь, что функция определена

```bash
flake8 telegram_bot.py tests/ --show-source
```

### Проблема: Сложность кода (C901)

**Решение**: Рассмотрите рефакторинг сложных функций

```bash
flake8 telegram_bot.py tests/ --statistics
```

### Проблема: Не хватает библиотек для mypy

**Решение**: Установите type stubs или используйте --ignore-missing-imports

```bash
mypy telegram_bot.py tests/ --ignore-missing-imports
```

## Полезные ссылки

- [PyTest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)