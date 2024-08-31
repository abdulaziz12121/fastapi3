from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

model = joblib.load('DBSCAN_model.joblib')
scaler = joblib.load('scaler.joblib')

app = FastAPI()

class InputFeatures(BaseModel):
    yellow:float
    red:float
    position_encoded:int
    
def preprocessing(input_features: InputFeatures):
    dict_f = {
    'yellow': input_features.yellow,
        'red': input_features.red,
    'position_encoded': input_features.position_encoded
     }
    feature_list = [dict_f[key] for key in sorted(dict_f)]
    return scaler.transform([feature_list])
 

@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

@app.get("/try/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.fit_predict(data)  
    return {"cluster": int(y_pred[0])}
