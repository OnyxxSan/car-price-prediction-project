from pathlib import Path
import re
import pandas as pd


def normalize_model_family(model_value: str) -> str:
    """Collapse trim-heavy model names into a stable family label."""
    raw = str(model_value).strip()
    if not raw:
        return ''

    # Keep letters, digits, spaces, and dashes; remove noisy punctuation.
    cleaned = re.sub(r'[^A-Za-z0-9\-\s]', ' ', raw)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return ''

    parts = cleaned.split(' ')
    if len(parts) == 1:
        return parts[0].title()

    # Preserve known two-token families such as "3 Series" and "Model S".
    if parts[0].isdigit() and parts[1].lower() in {'series'}:
        return f"{parts[0]} {parts[1].title()}"
    if parts[0].lower() == 'model' and len(parts[1]) <= 2:
        return f"Model {parts[1].upper()}"

    return parts[0].title()


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
    """Clean and standardize the used-car dataset for modeling."""
    cleaned = df.copy()

    cleaned.columns = cleaned.columns.str.strip().str.lower()

    def to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace(r'[^0-9.-]', '', regex=True),
            errors='coerce',
        )

    if 'price' in cleaned.columns:
        cleaned['price'] = to_numeric(cleaned['price'])
    if 'milage' in cleaned.columns:
        cleaned['milage'] = to_numeric(cleaned['milage'])

    cleaned = cleaned.dropna(subset=['price']).copy()

    if 'model_year' in cleaned.columns:
        current_year = pd.Timestamp.today().year
        cleaned['car_age'] = current_year - cleaned['model_year']

    categorical_cols = [
        'brand', 'model', 'fuel_type', 'engine', 'transmission',
        'ext_col', 'int_col', 'accident', 'clean_title'
    ]
    for col in categorical_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    if 'price' in cleaned.columns:
        cleaned = cleaned[cleaned['price'] > 0].copy()
    if 'milage' in cleaned.columns:
        cleaned = cleaned[cleaned['milage'] >= 0].copy()

    if 'model' in cleaned.columns:
        cleaned['model_family'] = cleaned['model'].apply(normalize_model_family)

    return cleaned
