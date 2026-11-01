from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "car_prices.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "car_price_model.joblib"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"


def generate_car_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 1800

    makes = ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Audi", "Volkswagen", "Nissan"]
    models = {
        "Toyota": ["Corolla", "Camry", "Prius", "RAV4"],
        "Honda": ["Civic", "Accord", "CR-V", "Fit"],
        "Ford": ["Focus", "Fiesta", "Mustang", "Explorer"],
        "BMW": ["3 Series", "5 Series", "X5", "X3"],
        "Mercedes": ["C Class", "E Class", "GLC", "GLE"],
        "Audi": ["A3", "A4", "Q5", "Q7"],
        "Volkswagen": ["Golf", "Passat", "Tiguan", "Polo"],
        "Nissan": ["Altima", "Sentra", "Juke", "Qashqai"],
    }
    fuel_choices = ["Petrol", "Diesel", "Hybrid", "Electric"]
    transmission_choices = ["Manual", "Automatic"]
    body_choices = ["Sedan", "SUV", "Hatchback", "Coupe"]

    make_values = rng.choice(makes, size=n_rows)
    model_values = [
        rng.choice(models.get(make, ["Model"])) for make in make_values
    ]
    year_values = rng.integers(2003, 2025, size=n_rows)
    mileage_values = rng.integers(2_000, 220_000, size=n_rows)
    fuel_values = rng.choice(fuel_choices, size=n_rows, p=[0.45, 0.30, 0.17, 0.08])
    transmission_values = rng.choice(transmission_choices, size=n_rows, p=[0.65, 0.35])
    body_values = rng.choice(body_choices, size=n_rows, p=[0.35, 0.25, 0.25, 0.15])
    engine_size_values = rng.choice([1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.0], size=n_rows)
    seats_values = rng.choice([2, 4, 5, 7], size=n_rows, p=[0.05, 0.60, 0.25, 0.10])

    # Create a realistic target variable based on common pricing factors.
    price_base = (
        5500
        + (year_values - 2000) * 900
        + (engine_size_values * 3000)
        + (seats_values * 200)
    )

    make_bias = {
        "Toyota": 4000,
        "Honda": 3500,
        "Ford": 2000,
        "BMW": 9000,
        "Mercedes": 11000,
        "Audi": 8500,
        "Volkswagen": 3000,
        "Nissan": 2500,
    }
    fuel_bias = {
        "Petrol": 0,
        "Diesel": 1800,
        "Hybrid": 3500,
        "Electric": 6000,
    }
    transmission_bias = {
        "Manual": 0,
        "Automatic": 2400,
    }
    body_bias = {
        "Sedan": 0,
        "SUV": 2800,
        "Hatchback": -1000,
        "Coupe": 1800,
    }

    price = (
        price_base
        + np.array([make_bias[m] for m in make_values])
        + np.array([fuel_bias[f] for f in fuel_values])
        + np.array([transmission_bias[t] for t in transmission_values])
        + np.array([body_bias[b] for b in body_values])
        - mileage_values * 0.18
        + rng.normal(0, 3200, size=n_rows)
    )

    data = pd.DataFrame(
        {
            "make": make_values,
            "model": model_values,
            "year": year_values,
            "mileage": mileage_values,
            "fuel_type": fuel_values,
            "transmission": transmission_values,
            "engine_size": engine_size_values,
            "body_type": body_values,
            "seats": seats_values,
            "price": np.round(price, 2),
        }
    )

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
    return data


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return generate_car_data()
    return pd.read_csv(DATA_PATH)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_model_pipeline(model_name: str) -> Pipeline:
    model_map = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", None),
            ("model", model_map[model_name]),
        ]
    )


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [col for col in data.columns if col != "price"]
    X = data[feature_columns]
    y = data["price"]
    return X, y


def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=42)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2,
    }


def save_model_artifacts(model: Pipeline, feature_columns: list[str]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_PATH, "w", encoding="utf-8") as file:
        json.dump(feature_columns, file)


def load_saved_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run python scripts/train_model.py first.")
    return joblib.load(MODEL_PATH)


def load_saved_feature_columns() -> list[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Feature metadata not found. Run python scripts/train_model.py first.")
    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
