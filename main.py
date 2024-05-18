from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

# Reemplace esto con su implementación:
class ApiInput(BaseModel):
    features: List[float]

# Reemplace esto con su implementación:
class ApiOutput(BaseModel):
    forecast:float

app = FastAPI()
model = joblib.load("model.joblib")

# Reemplace esto con su implementación:
@app.post("/predict")
async def predict(data: ApiInput) -> ApiOutput:
    import numpy as np
    ### ESCRIBA SU CÓDIGO AQUÍ ###
    model = joblib.load("model.joblib") # cargamos el modelo.
    predictions = model.predict(np.reshape((data.features),(1,5)))
    prediction = ApiOutput(forecast=predictions)
    return prediction
