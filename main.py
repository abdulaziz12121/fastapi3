from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load the saved KMeans model and scaler1
dbscan_model = joblib.load('DBSCAN_model1.joblib')
DBSCAN_scaler1 = joblib.load('scaler1_means1.joblib')

app = FastAPI()

class InputFeatures(BaseModel):
    yellow:float
    red:float
    position_encoded:int


def preprocess_dbscan(input_features: InputFeatures):
    data = [[input_features.yellow, input_features.red, input_features.position_encoded]]
    scaled_data = DBSCAN_scaler1.transform(data)
    return scaled_data


@app.post("/predict_dbscan")
async def predict_dbscan(input_features: InputFeatures):
    data = preprocess_dbscan(input_features)
    y_pred = dbscan_model.fit_predict(data)
    return {"dbscan_pred": int(y_pred[0])}


    

 



