# Сервис прогнозирования риска срыва доставки заказа

End-to-end учебный проект по курсу «Инженерия искусственного интеллекта».  
Предметная область: e-commerce и городская доставка.  
ML-задача: бинарная классификация `is_late_delivery` — предсказать, будет ли заказ доставлен с опозданием.

## Что внутри

- Синтетический датасет заказов: `data/delivery_orders_synthetic.csv`
- Генератор данных с управляемыми факторами риска: `src/data_generation.py`
- Обучение и сравнение `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`
- Сохранённая модель: `artifacts/model.joblib`
- Метаданные эксперимента и leaderboard: `artifacts/model_metadata.json`, `artifacts/leaderboard.json`
- FastAPI-сервис с `/health`, `/predict`, `/model-info`, `/metrics`
- Логи, базовые метрики, Dockerfile, `.env.example` и smoke-тесты

## Структура

- `src/` — генерация данных, обучение, схемы API, сервис
- `data/` — синтетические учебные данные
- `artifacts/` — модель, метаданные, таблица метрик, примеры заказов
- `configs/` — конфигурация обучения
- `notebooks/` — EDA и протокол экспериментов
- `tests/` — проверки API
- `report.md` — отчёт по проекту
- `self-checklist.md` — самопроверка перед сдачей

## Быстрый запуск

```bash
cd project
python -m pip install -r requirements.txt
python -m src.train
uvicorn src.service:app --host 0.0.0.0 --port 8000
```

После запуска:

- Swagger UI: `http://localhost:8000/docs`
- Health-check: `http://localhost:8000/health`
- Метрики: `http://localhost:8000/metrics`
- Информация о модели: `http://localhost:8000/model-info`

## Пример запроса

Файл с готовым телом запроса лежит в `src/demo_request.json`.

```bash
curl.exe -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  --data @src/demo_request.json
```

Пример ответа:

```json
{
  "model_name": "logistic_regression",
  "predictions": [
    {
      "is_late_delivery": 1,
      "late_probability": 1.0,
      "risk_level": "high",
      "recommended_action": "Переназначить курьера или предупредить клиента о риске задержки."
    },
    {
      "is_late_delivery": 0,
      "late_probability": 0.0097,
      "risk_level": "low",
      "recommended_action": "Стандартная обработка без ручной эскалации."
    }
  ]
}
```

## Обучение модели

```bash
cd project
python -m src.train
```

Скрипт:

1. генерирует 6500 синтетических заказов;
2. делит данные на train/test с сохранением баланса классов;
3. обучает три модели;
4. выбирает финальную модель по `F1`;
5. сохраняет модель и метаданные в `artifacts/`.

Фактические метрики на тестовой выборке:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.7638 | 0.6877 | 0.7647 | 0.7242 | 0.8365 |
| RandomForestClassifier | 0.7385 | 0.6632 | 0.7211 | 0.6909 | 0.8195 |
| GradientBoostingClassifier | 0.7592 | 0.7238 | 0.6565 | 0.6886 | 0.8237 |

Финальная модель: `LogisticRegression`. Она дала лучший `F1` и лучший `Recall`, что важно для сценария раннего предупреждения о проблемных доставках.

## Docker

```bash
cd project
docker build -t delivery-delay-risk-service .
docker run -p 8000:8000 delivery-delay-risk-service
```

Или:

```bash
cd project
docker compose up --build
```

## Тесты

```bash
cd project
python -m unittest tests/test_service.py
```

Тесты проверяют:

- `/health`;
- рабочий прогноз через `/predict`;
- отклонение некорректного payload;
- наличие базовых метрик.

## Сценарий демонстрации

1. Показать структуру проекта: `src/`, `data/`, `artifacts/`, `notebooks/`, `tests/`.
2. Открыть `notebooks/01_eda_and_experiments.ipynb` и показать EDA: баланс классов, влияние погоды, трафика и типа доставки.
3. Запустить `python -m src.train` и показать таблицу метрик.
4. Запустить API через `uvicorn src.service:app --host 0.0.0.0 --port 8000`.
5. В Swagger UI отправить `src/demo_request.json` в `/predict`.
6. Показать `/metrics` и `/model-info`.

## Ограничения

- Данные синтетические, поэтому модель демонстрирует инженерный цикл, а не реальное качество в продакшене.
- Нет базы данных, авторизации и интеграции с реальной системой маршрутизации.
- Порог решения фиксирован на `0.5`; в реальном бизнес-сценарии его стоит подбирать по стоимости ошибок.
