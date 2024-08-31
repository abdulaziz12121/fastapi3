from typing import Optional
import joblib
from pydantic import BaseModel
from fastapi import FastAPI

model = joblib.load('DBSCAN_model.joblib')
# scaler = joblib.load('Models/scaler.joblib')

# model=pickle.load(open('train_model.sav','rb'))

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}