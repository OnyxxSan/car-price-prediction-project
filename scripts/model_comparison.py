from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.car_price_pipeline import (
    build_model_pipeline,
    evaluate_model,
    load_dataset,
    make_preprocessor,
    split_features_target,
    train_test_split_data,
)


def run_model_comparison() -> None:
    data = load_dataset()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)

    models = {
        "Linear Regression": "linear_regression",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
    }

    print("Starting model comparison...\n")
    print(f"Dataset rows: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print("\nModel Performance:\n")
    print(f"{'Model':<20} {'MAE':>12} {'RMSE':>12} {'R2':>10}")
    print("-" * 62)

    results = {}
    for model_name, model_key in models.items():
        pipeline = build_model_pipeline(model_key)
        pipeline.steps[0] = ("preprocessor", make_preprocessor(X_train))
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[model_name] = metrics
        print(
            f"{model_name:<20} {metrics['MAE']:>12.2f} {metrics['RMSE']:>12.2f} {metrics['R2']:>10.4f}"
        )

    best_model_name = max(results, key=lambda name: results[name]["R2"])
    best_model = results[best_model_name]
    print("\nBest model by R2:", best_model_name)
    print(
        f"MAE={best_model['MAE']:.2f}, RMSE={best_model['RMSE']:.2f}, R2={best_model['R2']:.4f}"
    )


if __name__ == "__main__":
    run_model_comparison()
