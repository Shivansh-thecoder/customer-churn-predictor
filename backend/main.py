from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

MODEL_PATH=os.path.join(os.path.dirname(__file__),'..','models','rf_model.pkl')
with open(MODEL_PATH,'rb') as f:
    model=pickle.load(f)

app=FastAPI(title="Customer Churn Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "FastAPI backend is running!"}


@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # Convert Pydantic model to DataFrame
        input_df = pd.DataFrame([data.model_dump()])  # preserves column names

        # Predict using pipeline
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {"prediction": int(prediction), "probability": float(probability)}

    except Exception as e:
        return {"error": str(e)}

