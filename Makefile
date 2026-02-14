# Makefile для Protocolista Telegram Bot
# Упрощает выполнение типичных задач

.PHONY: help install install-dev lint lint-fix test test-cov format check clean

help: ## Показать все доступные команды
	@echo "Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Установить основные зависимости
	pip install -r requirements.txt

install-dev: ## Установить все зависимости для разработки
	pip install -r requirements-dev.txt
	pre-commit install

lint: ## Запустить все линтеры
	flake8 telegram_bot.py
	mypy telegram_bot.py
	bandit -r telegram_bot.py
	@echo "Все линтеры успешно пройдены!"

lint-fix: ## Запустить автоисправление линтеров
	isort telegram_bot.py
	black telegram_bot.py
	@echo "Код отформатирован!"

test: ## Запустить тесты
	pytest tests/ -v

test-cov: ## Запустить тесты с покрытием
	pytest tests/ --cov=telegram_bot --cov-report=term-missing --cov-report=html

test-unit: ## Запустить только unit тесты
	pytest tests/ -m unit -v

test-integration: ## Запустить только integration тесты
	pytest tests/ -m integration -v

check: ## Полная проверка (lint + test)
	lint
	test

format: ## Отформатировать код
	isort telegram_bot.py
	black telegram_bot.py
	@echo "Код отформатирован с помощью Black и isort!"

clean: ## Очистить временные файлы
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf __pycache__
	rm -rf htmlcov
	rm -rf .coverage
	rm -f *.pyc *.pyo
	@echo "Временные файлы удалены!"

run: ## Запустить бота
	python telegram_bot.py

install-deps: ## Установить все зависимости (основные + dev)
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

ci: ## Запуск для CI/CD
	lint-fix
	test-cov
	@echo "CI успешно завершен!"