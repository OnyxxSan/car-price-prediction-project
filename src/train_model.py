from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_preprocessing import clean_dataset, load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'used_car_model.joblib'


def build_model_pipeline() -> Pipeline:
    """Return a scikit-learn pipeline for the car price model."""
    numeric_features = ['model_year', 'milage', 'car_age']
    categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'accident', 'clean_title']

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
            ('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    return pipeline


def train_and_evaluate() -> dict:
    """Load, preprocess, train, evaluate, and save the best model."""
    df = load_dataset()
    df = clean_dataset(df)

    feature_columns = ['brand', 'model', 'model_year', 'milage', 'fuel_type', 'transmission', 'accident', 'clean_title', 'car_age']
    X = df[feature_columns]
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
    metrics = {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': mean_squared_error(y_test, predictions, squared=False),
        'r2': r2_score(y_test, predictions),
    }

    joblib.dump(pipeline, MODEL_PATH)

    print('Model saved to:', MODEL_PATH)
    print('MAE:', round(metrics['mae'], 2))
    print('RMSE:', round(metrics['rmse'], 2))
    print('R²:', round(metrics['r2'], 4))

    return metrics


if __name__ == '__main__':
    train_and_evaluate()
