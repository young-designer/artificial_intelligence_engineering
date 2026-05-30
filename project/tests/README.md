# tests

Smoke-тесты FastAPI-сервиса.

Запуск:

```bash
python -m unittest tests/test_service.py
```

Проверяются `/health`, `/predict`, `/metrics` и валидация некорректного запроса.
