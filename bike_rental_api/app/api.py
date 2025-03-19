import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from bike_rental_model import __version__ as model_version
from bike_rental_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()

example_input = {

    "inputs" : [
        {
            "dteday": "2012-05-13",
            "season": "summer",
            "hr": "12pm",
            "holiday": "No",
            "weekday": "Sun",
            "workingday": "No",
            "weathersit": "Clear",
            "temp": 22.08,
            "atemp": 24.99,
            "hum": 56.99,
            "windspeed": 15.00,
            "casual": 189,
            "registered": 342,
        }
    ]
}

@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    predictions with the bike_rental_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results

