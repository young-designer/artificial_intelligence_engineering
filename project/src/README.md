# src

Основной код проекта.

- `data_generation.py` — генерация синтетических заказов доставки.
- `train.py` — обучение моделей, сравнение метрик, сохранение артефактов.
- `service.py` — FastAPI-сервис для инференса.
- `schemas.py` — Pydantic-схемы входа и выхода API.
- `features.py` — подготовка входного JSON к формату модели.
- `config.py` — загрузка `.env` и `configs/train_config.yaml`.
- `logging_utils.py` — единая настройка логирования.
- `demo_request.json` — готовый пример запроса к `/predict`.

Основные команды:

```bash
python -m src.train
uvicorn src.service:app --host 0.0.0.0 --port 8000
```
