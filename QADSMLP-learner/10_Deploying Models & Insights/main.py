from typing import Union
from utilities import make_prediction
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/{item_id}")
def get_prediction(item_id: int, ds: Union[str], fr: Union[str], vc: Union[str]):
    data = {"destination": [ds], "fare": [fr], "vehicle_class": [vc]}
    return {"item_id": item_id, "model_in":data, "model_out": round(make_prediction(data)[0], 2)}
