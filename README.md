# Predictive Maintenance for Equipment

This project aims to predict equipment failures using machine learning techniques. The model is trained on sensor data from various equipment.

## Project Structure

- **data/**: Contains raw, processed, and external data files.
- **notebooks/**: Jupyter notebooks for data exploration, preprocessing, and model training.
- **src/**: Source code for data preprocessing, feature engineering, and model training.
- **models/**: Trained model files.
- **results/**: Evaluation results.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run data preprocessing: `python src/data_preprocessing.py`
3. Run feature engineering: `python src/feature_engineering.py`
4. Train the model: `python src/model.py`

## Dependencies

- pandas
- matplotlib
- scikit-learn
- joblib
