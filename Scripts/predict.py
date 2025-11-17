import os
import pandas as pd
import joblib
from config import PROJECT_ROOT

# Paths
model_dir = os.path.join(PROJECT_ROOT, 'models')
model_path = os.path.join(model_dir, 'churn_model.pkl')
preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
columns_path = os.path.join(model_dir, 'feature_columns.pkl')

# Load the model, preprocessor, and feature columns
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
feature_columns = joblib.load(columns_path)

def predict_churn(customer_data):
    """
    Predict churn for a given customer data.

    Parameters:
    customer_data (dict): A dictionary containing customer features.

    Returns:
    Churn Probability and Prediction (str): The probability of churn and the prediction label.
    """
    # Convert input data to DataFrame
    input_df = pd.DataFrame([customer_data])

    # Ensure all feature columns are present
    df = pd.reindex(columns=feature_columns, fill_value=0)

    # Scake numeric features
    df_scaled = preprocessor.transform(df)

    # Predict churn
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]

    return {
        "Churn Probability": float(probability),
        "Prediction": int(prediction)
    }

if __name__ == "__main__":
    result = predict_churn()
