# MNIST

Классификация рукописных цифр (датасет MNIST) — softmax-регрессия на чистом NumPy.

## Структура

- `mnist.py` — основной скрипт обучения
- `mnist.ipynb` — Jupyter-ноутбук с тем же кодом и выводами
- `pyproject.toml` — зависимости проекта (uv)

## Стек

- Python 3.14+, NumPy, scikit-learn (только для загрузки данных и train/test split), pandas
- Менеджер пакетов: uv

## Модель

Линейный классификатор (softmax-регрессия), обучается mini-batch SGD.
- Вход: 784 признака (28x28 пикселей, нормализованных в [0,1])
- Выход: 10 классов (цифры 0-9)
- Loss: cross-entropy
- Гиперпараметры: lr=0.1, epochs=200, batch_size=256
- Результат: ~92% accuracy на тесте

## Запуск

```bash
uv run mnist.py
```
