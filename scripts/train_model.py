from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.car_price_pipeline import (
    MODELS_DIR,
    build_model_pipeline,
    evaluate_model,
    load_dataset,
    make_preprocessor,
    save_model_artifacts,
    split_features_target,
    train_test_split_data,
)


def train_and_save_model() -> None:
    data = load_dataset()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)

    model_name = "random_forest"
    pipeline = build_model_pipeline(model_name)
    pipeline.steps[0] = ("preprocessor", make_preprocessor(X_train))
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    print("Training complete.")
    print(f"Model used: {model_name}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2: {metrics['R2']:.4f}")

    feature_columns = X.columns.tolist()
    save_model_artifacts(pipeline, feature_columns)
    print(f"Saved model to: {MODELS_DIR / 'car_price_model.joblib'}")


if __name__ == "__main__":
    train_and_save_model()
