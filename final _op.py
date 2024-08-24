import pandas as pd
import joblib
import numpy as np

# Load trained models
rf_model = joblib.load('random_forest_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')

def predict_health_condition(signal_value, soil_moisture, temperature):
    # Prepare data for prediction with feature names
    X = pd.DataFrame([[signal_value, soil_moisture, temperature]], columns=['signal', 'soil_moisture', 'temperature'])

    # Predict with Random Forest Classifier
    rf_prediction = rf_model.predict(X)[0]

    # Predict with Logistic Regression
    lr_prediction = lr_model.predict(X)[0]

    return rf_prediction, lr_prediction

if __name__ == "__main__":
    # Manually input sensor readings from terminal
    signal_value = float(input("Enter signal value: "))
    soil_moisture = float(input("Enter soil moisture (%): "))
    temperature = float(input("Enter temperature (Celsius): "))

    # Predict health condition
    rf_pred, lr_pred = predict_health_condition(signal_value, soil_moisture, temperature)

    # Display results in terminal
    print(f"Random Forest Prediction: {rf_pred}")
    print(f"Logistic Regression Prediction: {lr_pred}")


