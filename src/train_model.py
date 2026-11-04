from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_preprocessing import clean_dataset, load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'used_car_model.joblib'
FEATURE_COLUMNS = [
    'brand',
    'model',
    'model_year',
    'milage',
    'fuel_type',
    'transmission',
    'accident',
    'clean_title',
    'car_age',
]


def build_model_pipeline() -> Pipeline:
    """Return the final model pipeline used for production predictions."""
    numeric_features = ['model_year', 'milage', 'car_age']
    categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'accident', 'clean_title']

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
            ('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_features),
        ]
    )

    model = LinearRegression()
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])


def compute_baseline_metrics(y_true: pd.Series) -> dict:
    """Compute a simple baseline based on the median target value."""
    median_price = y_true.median()
    baseline_predictions = [median_price] * len(y_true)
    return {
        'mae': mean_absolute_error(y_true, baseline_predictions),
        'rmse': mean_squared_error(y_true, baseline_predictions, squared=False),
        'r2': r2_score(y_true, baseline_predictions),
    }


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
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    test_metrics = {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': mean_squared_error(y_test, predictions, squared=False),
        'r2': r2_score(y_test, predictions),
    }

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring='r2',
        n_jobs=-1,
    )

    baseline_metrics = compute_baseline_metrics(y_test)

    model_dir = MODEL_PATH.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print('Model saved to:', MODEL_PATH)
    print(f"MAE: {test_metrics['mae']:.2f}")
    print(f"RMSE: {test_metrics['rmse']:.2f}")
    print(f"R²: {test_metrics['r2']:.4f}")
    print(f"CV R² mean: {cv_scores.mean():.4f}")
    print(f"Baseline MAE: {baseline_metrics['mae']:.2f}")

    return {
        'test_metrics': test_metrics,
        'cv_r2_mean': float(cv_scores.mean()),
        'baseline_metrics': baseline_metrics,
        'model_path': str(MODEL_PATH),
    }


if __name__ == '__main__':
    train_and_evaluate()
