from pathlib import Path
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_preprocessing import clean_dataset, load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'used_car_model.joblib'
FEATURE_COLUMNS = [
    'brand',
    'model_family',
    'model_year',
    'milage',
    'fuel_type',
    'transmission',
    'accident',
    'clean_title',
    'car_age',
]


def build_model_pipeline() -> TransformedTargetRegressor:
    """Return the production model with preprocessing and target transform."""
    numeric_features = ['model_year', 'milage', 'car_age']
    categorical_features = ['brand', 'model_family', 'fuel_type', 'transmission', 'accident', 'clean_title']

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
            ('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20)),
            ]), categorical_features),
        ]
    )

    regressor = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def compute_regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict:
    """Compute common regression metrics on the original target scale."""
    mse = mean_squared_error(y_true, predictions)
    return {
        'mae': mean_absolute_error(y_true, predictions),
        'rmse': sqrt(mse),
        'r2': r2_score(y_true, predictions),
        'mape': mean_absolute_percentage_error(y_true, predictions),
    }


def compute_price_segment_mae(
    y_true: pd.Series,
    predictions: np.ndarray,
    *,
    upper_bound: float,
) -> float | None:
    """Return MAE for cars priced at or below upper_bound on the test split."""
    mask = y_true <= upper_bound
    if not mask.any():
        return None
    return float(mean_absolute_error(y_true[mask], predictions[mask]))


def compute_training_sample_weights(y_train: pd.Series) -> np.ndarray:
    """Up-weight lower-priced training samples to reduce low-price miss magnitude."""
    # Use inverse sqrt weighting so low-price cars matter more without exploding weights.
    price_values = np.clip(y_train.to_numpy(dtype=float), 1000.0, None)
    weights = 1.0 / np.sqrt(price_values)
    return weights / weights.mean()


def compute_baseline_metrics(y_true: pd.Series, train_target_median: float) -> dict:
    """Compute baseline metrics on y_true using the train-set median as predictor."""
    baseline_predictions = np.full(shape=len(y_true), fill_value=train_target_median, dtype=float)
    return compute_regression_metrics(y_true, baseline_predictions)


def train_and_evaluate() -> dict:
    """Load, clean, train, evaluate, and save the final model."""
    df = load_dataset()
    df = clean_dataset(df)

    X = df[FEATURE_COLUMNS]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_model_pipeline()
    training_sample_weights = compute_training_sample_weights(y_train)
    pipeline.fit(X_train, y_train, model__sample_weight=training_sample_weights)

    predictions = pipeline.predict(X_test)
    predictions = np.clip(predictions, 0.0, None)
    test_metrics = compute_regression_metrics(y_test, predictions)
    low_price_mae = compute_price_segment_mae(y_test, predictions, upper_bound=15000.0)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring='r2',
        n_jobs=-1,
    )

    baseline_train_median = float(y_train.median())
    baseline_metrics = compute_baseline_metrics(y_test, baseline_train_median)

    model_dir = MODEL_PATH.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print('Model saved to:', MODEL_PATH)
    print(f"MAE: {test_metrics['mae']:.2f}")
    print(f"RMSE: {test_metrics['rmse']:.2f}")
    print(f"R²: {test_metrics['r2']:.4f}")
    print(f"MAPE: {test_metrics['mape'] * 100:.2f}%")
    print(f"CV R² mean: {cv_scores.mean():.4f}")
    print(f"Baseline MAE: {baseline_metrics['mae']:.2f}")
    print(f"Baseline RMSE: {baseline_metrics['rmse']:.2f}")
    print(f"Baseline R²: {baseline_metrics['r2']:.4f}")
    print(f"Baseline MAPE: {baseline_metrics['mape'] * 100:.2f}%")
    if low_price_mae is not None:
        print(f"Low-price MAE (<= $15k): {low_price_mae:.2f}")

    return {
        'test_metrics': test_metrics,
        'cv_r2_mean': float(cv_scores.mean()),
        'baseline_metrics': baseline_metrics,
        'low_price_mae_le_15k': low_price_mae,
        'model_path': str(MODEL_PATH),
    }


if __name__ == '__main__':
    train_and_evaluate()
