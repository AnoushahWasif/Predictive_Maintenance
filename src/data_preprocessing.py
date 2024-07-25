import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file):
    # Load raw data
    data = pd.read_csv(input_file)

    # Convert date to datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Handle missing values by forward filling
    data.fillna(method='ffill', inplace=True)

    # Drop duplicates
    data.drop_duplicates(inplace=True)

    # Standardize the metrics
    metrics = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']
    scaler = StandardScaler()
    data[metrics] = scaler.fit_transform(data[metrics])

    # Save the processed data
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    preprocess_data('data/raw/equipment_data.csv', 'data/processed/equipment_data_processed.csv')

