from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, xgboost as xgb, pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Load model & preprocessing files
model = xgb.XGBClassifier()
model.load_model(str(BASE_DIR / "cholera_xgb_model.json"))

pre = joblib.load(str(BASE_DIR / "cholera_preprocess.pkl"))
feature_columns = joblib.load(str(BASE_DIR / "feature_columns.pkl"))

label_encoders = pre["label_encoders"]
le_risk = pre["risk_encoder"]
categorical_cols = pre["categorical_cols"]

app = FastAPI(title="Cholera Risk Prediction API")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this to ["https://nilevalleyuniversity.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: int
    gender: str
    region: str
    clean_water: str
    toilet: str
    vomiting: int
    diarrhea: int
    stomach_pain: int
    fatigue: int
    travel_history: str
    fever: int

def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    for col in categorical_cols:
        le = label_encoders[col]
        df[col] = df[col].astype(str).str.lower()
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

@app.get("/")
def root():
    return {"message": "Cholera Risk Prediction API is running."}

@app.post("/predict")
def predict(data: PatientData):
    processed = preprocess_input(data.dict())
    pred = model.predict(processed)[0]
    risk_level = le_risk.inverse_transform([pred])[0]
    return {"predicted_risk": risk_level}
