from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.car_price_pipeline import generate_car_data

if __name__ == "__main__":
    generate_car_data()
    print("Dataset created successfully at data/car_prices.csv")
