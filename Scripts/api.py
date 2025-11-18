import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from config import PROJECT_ROOT

#Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Machine learning API for predicting customer churn .",
    version="1.0.0"
)

# Load model files
model_dir = os.path.join(PROJECT_ROOT, 'models')
model_path = os.path.join(model_dir, 'churn_model.pkl')
preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
columns_path = os.path.join(model_dir, 'feature_columns.pkl')

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
feature_columns = joblib.load(columns_path)

# Define pydantic model for input data
class CustomerInput(BaseModel):
    from pydantic import BaseModel

    class CustomerInput(BaseModel):
        SeniorCitizen: int
        tenure: int
        MonthlyCharges: float
        TotalCharges: float

        MultipleLines_No_phone_service: int
        MultipleLines_Yes: int

        InternetService_Fiber_optic: int
        InternetService_No: int

        OnlineSecurity_No_internet_service: int
        OnlineSecurity_Yes: int

        OnlineBackup_No_internet_service: int
        OnlineBackup_Yes: int

        DeviceProtection_No_internet_service: int
        DeviceProtection_Yes: int

        TechSupport_No_internet_service: int
        TechSupport_Yes: int

        StreamingTV_No_internet_service: int
        StreamingTV_Yes: int

        StreamingMovies_No_internet_service: int
        StreamingMovies_Yes: int

        Contract_One_year: int
        Contract_Two_year: int

        PaymentMethod_Credit_card_automatic: int
        PaymentMethod_Electronic_check: int
        PaymentMethod_Mailed_check: int

        gender_Male: int
        Partner_Yes: int
        Dependents_Yes: int
        PhoneService_Yes: int
        PaperlessBilling_Yes: int


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "API is running", "model_loaded": True}

# Prediction endpoint
@app.post("/predict")
def predict_churn(input_data: CustomerInput):

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Ensure all feature columns are present
    df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric features
    df_scaled = preprocessor.transform(df)

    # Predict churn
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]

    return {
        "Probability": int(probability),
        "Prediction": int(prediction)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

