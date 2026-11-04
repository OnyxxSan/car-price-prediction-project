from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'used_car_model.joblib'


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError('Model not found. Train it first using: python src/train_model.py')
    return joblib.load(MODEL_PATH)


def build_feature_dict(sample: dict) -> dict:
    """Convert user input into a DataFrame-compatible feature dictionary."""
    current_year = pd.Timestamp.today().year

    row = {
        'brand': str(sample.get('brand', '')).strip(),
        'model': str(sample.get('model', '')).strip(),
        'model_year': int(sample.get('model_year', 0)),
        'milage': float(sample.get('milage', 0)),
        'fuel_type': str(sample.get('fuel_type', '')).strip(),
        'transmission': str(sample.get('transmission', '')).strip(),
        'accident': str(sample.get('accident', '')).strip(),
        'clean_title': str(sample.get('clean_title', '')).strip(),
        'car_age': max(0, current_year - int(sample.get('model_year', 0))),
    }

    return row


def predict_price(sample: dict) -> float:
    model = load_model()
    row = pd.DataFrame([build_feature_dict(sample)])
    prediction = model.predict(row)
    return float(prediction[0])
