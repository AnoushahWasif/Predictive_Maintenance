import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(data_file, model_file, report_file):
    data = pd.read_csv(data_file)

    # Define features and target
    X = data.drop(['date', 'device', 'failure'], axis=1)
    y = data['failure']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_file)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save evaluation results
    with open(report_file, 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(report)

if __name__ == "__main__":
    train_model('data/processed/equipment_data_features.csv', 'models/model.pkl', 'results/evaluation_report.txt')
