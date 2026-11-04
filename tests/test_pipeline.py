import pandas as pd

from src.data_preprocessing import clean_dataset, load_dataset
from src.predict import build_feature_dict


def test_load_dataset_file_exists():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'price' in df.columns


def test_clean_dataset_adds_car_age_and_removes_invalid_prices():
    df = pd.DataFrame({
        'brand': ['Toyota', 'Honda'],
        'model': ['Corolla', 'Civic'],
        'model_year': [2018, 2020],
        'milage': ['45000', '51000'],
        'price': ['18000', '22000'],
        'fuel_type': ['Gasoline', 'Hybrid'],
        'transmission': ['Automatic', 'Manual'],
        'accident': ['None reported', 'None reported'],
        'clean_title': ['Yes', 'Yes'],
    })

    cleaned = clean_dataset(df)

    assert 'car_age' in cleaned.columns
    assert cleaned['car_age'].ge(0).all()
    assert cleaned['milage'].dtype.kind in {'i', 'f'}
    assert cleaned['price'].gt(0).all()


def test_build_feature_dict_creates_expected_values():
    sample = {
        'brand': 'Toyota',
        'model': 'Corolla',
        'model_year': 2020,
        'milage': 45000,
        'fuel_type': 'Gasoline',
        'transmission': 'Automatic',
        'accident': 'None reported',
        'clean_title': 'Yes',
    }

    row = build_feature_dict(sample)

    assert row['brand'] == 'Toyota'
    assert row['model_year'] == 2020
    assert row['milage'] == 45000.0
    assert row['car_age'] >= 0
