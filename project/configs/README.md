# configs

- `train_config.yaml` — конфигурация генерации данных, списка признаков, моделей и метрики выбора финальной модели.

Главные параметры:

- `data.rows` — размер синтетического датасета;
- `features.numeric/categorical/boolean` — признаки, используемые моделью;
- `training.selection_metric` — метрика выбора лучшей модели;
- `training.decision_threshold` — порог бинарного решения в API.
