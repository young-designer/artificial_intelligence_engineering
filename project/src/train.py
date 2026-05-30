from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROJECT_ROOT, get_settings, load_train_config
from src.data_generation import save_dataset
from src.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)


def feature_columns(config: dict) -> list[str]:
    return config["features"]["numeric"] + config["features"]["categorical"] + config["features"]["boolean"]


def build_preprocessor(config: dict) -> ColumnTransformer:
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
    boolean_features = config["features"]["boolean"]

    numeric_steps = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_steps, numeric_features),
            ("cat", categorical_steps, categorical_features),
            ("bool", "passthrough", boolean_features),
        ]
    )


def build_candidate_models(config: dict) -> dict[str, object]:
    model_config = config["training"]["models"]
    return {
        "logistic_regression": LogisticRegression(**model_config["logistic_regression"]),
        "random_forest": RandomForestClassifier(**model_config["random_forest"]),
        "gradient_boosting": GradientBoostingClassifier(**model_config["gradient_boosting"]),
    }


def evaluate_model(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predicted_labels = model.predict(features)
    predicted_probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": round(accuracy_score(target, predicted_labels), 4),
        "precision": round(precision_score(target, predicted_labels, zero_division=0), 4),
        "recall": round(recall_score(target, predicted_labels, zero_division=0), 4),
        "f1": round(f1_score(target, predicted_labels, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(target, predicted_probabilities), 4),
    }


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    config = load_train_config()

    data_config = config["data"]
    random_seed = int(config["project"]["random_seed"])
    dataset_path = PROJECT_ROOT / data_config["raw_dataset_path"]
    save_dataset(dataset_path, rows=int(data_config["rows"]), random_seed=random_seed)
    LOGGER.info("Dataset generated: %s", dataset_path)

    dataframe = pd.read_csv(dataset_path)
    target_column = data_config["target_column"]
    all_features = feature_columns(config)
    x = dataframe[all_features]
    y = dataframe[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(config["training"]["test_size"]),
        random_state=random_seed,
        stratify=y,
    )

    trained_models: dict[str, Pipeline] = {}
    leaderboard: dict[str, dict[str, float]] = {}

    for model_name, estimator in build_candidate_models(config).items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(config)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)
        trained_models[model_name] = pipeline
        leaderboard[model_name] = metrics
        LOGGER.info("%s: %s", model_name, metrics)

    selection_metric = config["training"]["selection_metric"]
    best_model_name = max(leaderboard, key=lambda name: leaderboard[name][selection_metric])
    best_model = trained_models[best_model_name]

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, settings.model_path)

    class_balance = dataframe[target_column].value_counts(normalize=True).sort_index().round(4).to_dict()
    metadata = {
        "project_name": config["project"]["name"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(dataframe)),
        "target_column": target_column,
        "id_column": data_config["id_column"],
        "feature_columns": all_features,
        "numeric_features": config["features"]["numeric"],
        "categorical_features": config["features"]["categorical"],
        "boolean_features": config["features"]["boolean"],
        "class_balance": {str(key): float(value) for key, value in class_balance.items()},
        "selection_metric": selection_metric,
        "decision_threshold": float(config["training"]["decision_threshold"]),
        "risk_thresholds": {"medium": 0.38, "high": 0.68},
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "best_metrics": leaderboard[best_model_name],
    }
    with settings.model_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    artifacts_dir = settings.model_path.parent
    with (artifacts_dir / "leaderboard.json").open("w", encoding="utf-8") as file:
        json.dump(leaderboard, file, indent=2, ensure_ascii=False)

    sample_columns = [data_config["id_column"]] + all_features + [target_column]
    dataframe[sample_columns].head(12).to_csv(artifacts_dir / "sample_orders.csv", index=False)
    LOGGER.info("Selected model: %s", best_model_name)


if __name__ == "__main__":
    main()
