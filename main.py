from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from typing import List


app = FastAPI()

# Load the saved DBSCAN model and scaler
dbscan_model = joblib.load('DBSCAN_model1.joblib')

class InputFeatures(BaseModel):
    yellow: float
    red: float
    position_encoded: int

def preprocess_dbscan(input_features: InputFeatures):
    data = [[input_features.yellow, input_features.red, input_features.position_encoded]]
    predictions = dbscan_model.fit_predict(data)
    return predictions

@app.post("/predict_dbscan")
async def predict_dbscan(input_features: InputFeatures):
    data = preprocess_dbscan(input_features)
    return {"dbscan_pred": int(data)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
