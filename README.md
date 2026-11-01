# Used Car Price Prediction

A portfolio-ready machine learning project that predicts used car prices using vehicle attributes such as brand, model, year, mileage, fuel type, transmission, and title status.

## Overview

This project applies a complete data science workflow to a real-world business problem: estimating the fair market value of a used car. Buyers and sellers often struggle with inconsistent pricing across listings, so this project creates a practical pricing estimator based on historical data.

The app is built with Python and Streamlit, allowing users to enter car details and receive an estimated market price in seconds.

## Problem and Why It Matters

Used car prices are influenced by a mix of factors, including age, mileage, condition, fuel efficiency, transmission type, and accident history. These details vary from listing to listing, which makes pricing difficult without a consistent model.

This project helps solve that challenge by building a predictive model that estimates car value based on real listing data. It is useful for:

- buyers who want a better sense of fair pricing
- sellers who want a benchmark for listing price
- portfolio projects that show practical machine learning skills

## Dataset

The project uses a structured dataset (`used_cars.csv`) containing vehicle information and a target variable, `price`.

### Features used in the model

- brand
- model
- model year
- mileage
- fuel type
- transmission
- accident status
- clean title
- vehicle age

### Target variable

- price: the used car selling price in dollars

The dataset is cleaned and prepared before training so that missing values are handled and the model can learn from consistent inputs.

## Machine Learning Workflow

The project follows a standard end-to-end workflow:

1. Data loading and inspection
   - load the raw dataset
   - confirm key columns and file structure

2. Data cleaning
   - standardize column names
   - convert numeric values to the correct format
   - remove invalid rows and missing values
   - create derived features such as vehicle age

3. Feature preparation
   - keep the most relevant attributes for price prediction
   - convert categorical features such as brand and fuel type into a usable format for machine learning

4. Train-test split
   - split the data into training and testing sets with a fixed random seed
   - evaluate how well the model generalizes to unseen data

5. Model comparison
   - train and compare multiple regression models
   - evaluate each model using MAE, RMSE, and R²

6. Model selection and saving
   - choose the best-performing model
   - save the trained model for future predictions

7. Prediction app
   - build a Streamlit interface for interactive use
   - allow users to enter vehicle details and get a price estimate

## Models Used

The project benchmarks several regression models:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The final model selection is based on performance on the test set.

## Results

The model comparison results on this dataset are:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 2479.42 | 3138.45 | 0.9435 |
| Random Forest Regressor | 3360.20 | 4134.82 | 0.9020 |
| Gradient Boosting Regressor | 2823.97 | 3459.01 | 0.9314 |

The best-performing model on this dataset was Linear Regression, with an R² of 0.9435. This means the model explains approximately 94.35% of the variation in used car prices based on the available feature set.

## Tools and Libraries

This project uses:

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Streamlit
- joblib
- Jupyter Notebook

## Project Structure

```text
car-price-prediction-project/
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_comparison.py
│   ├── predict.py
│   └── train_model.py
├── DS_Car_Price_Project.ipynb
├── README.md
├── requirements.txt
├── used_cars.csv
└── .gitignore
```

## Setup Instructions

### 1. Install Python

Download and install Python 3.10+ from the official Python website.

### 2. Open the project in VS Code

- open VS Code
- select the project folder
- open the integrated terminal

### 3. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Run Instructions

### 1. Ensure the dataset is available

The project expects the dataset file `used_cars.csv` in the project root or in the `data/raw` folder. The preprocessing script automatically looks for it.

### 2. Compare models

```bash
python src/model_comparison.py
```

This script trains and evaluates multiple regression models and prints the MAE, RMSE, and R² values for each one.

### 3. Train the final model

```bash
python src/train_model.py
```

This script builds the selected model pipeline, evaluates it, and saves the trained model to the `models/` directory.

### 4. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app opens a user-friendly interface where you can enter vehicle details and receive an estimated price.

## Streamlit App Screenshot

Save the app screenshot as `app/screenshots/streamlit_app.png` in the repo and it will display here:

![Streamlit app screenshot](app/screenshots/streamlit_app.png)

## Resume-Friendly Description

Developed a machine learning project to predict used car prices using vehicle attributes such as mileage, age, fuel type, transmission, and accident history. Built a clean data preprocessing pipeline, compared multiple regression models, and packaged the final model into an interactive Streamlit app for user-friendly price estimation.

## Interview-Friendly Description

This project shows my end-to-end data science workflow: cleaning and preparing real-world automotive data, training multiple regression models, comparing their performance, and selecting the strongest approach for a business use case. I also created a Streamlit interface so the model can be used interactively by non-technical users.

## Conclusion

This project combines machine learning, practical data cleaning, and application development in a single portfolio-ready workflow. It is designed to be easy to understand for recruiters, internship reviewers, and hiring managers.

## Notes

The original exploratory notebook, `DS_Car_Price_Project.ipynb`, is kept in the repository for reference, while the organized codebase makes the project easier to present and maintain.
