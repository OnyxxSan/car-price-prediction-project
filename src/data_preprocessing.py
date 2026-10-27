from pathlib import Path
import re
import pandas as pd


def load_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the car dataset from either the raw data folder or the repo root."""
    root = Path(__file__).resolve().parents[1]

    candidates = []
    if csv_path is not None:
        candidates.append(Path(csv_path))

    candidates.extend([
        root / 'data' / 'raw' / 'used_cars.csv',
        root / 'used_cars.csv',
    ])

    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)

    raise FileNotFoundError('Could not find used_cars.csv in the repository.')


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw used car dataset and prepare it for modeling."""
    cleaned = df.copy()

    # Standardize column names
    cleaned.columns = cleaned.columns.str.strip().str.lower()

    # Numeric conversion helpers
    def to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace(r'[^0-9.-]', '', regex=True),
            errors='coerce',
        )

    # Clean known price and mileage fields
    if 'price' in cleaned.columns:
        cleaned['price'] = to_numeric(cleaned['price'])
    if 'milage' in cleaned.columns:
        cleaned['milage'] = to_numeric(cleaned['milage'])

    # Keep only rows with non-null target value
    cleaned = cleaned.dropna(subset=['price']).copy()

    # Add a useful feature: age of vehicle
    if 'model_year' in cleaned.columns:
        current_year = pd.Timestamp.today().year
        cleaned['car_age'] = current_year - cleaned['model_year']

    # Normalize categorical columns
    categorical_cols = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
    for col in categorical_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    # Remove impossible or duplicate values if present
    if 'price' in cleaned.columns:
        cleaned = cleaned[cleaned['price'] > 0].copy()
    if 'milage' in cleaned.columns:
        cleaned = cleaned[cleaned['milage'] >= 0].copy()

    return cleaned
