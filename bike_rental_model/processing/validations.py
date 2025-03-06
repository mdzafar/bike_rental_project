import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from datetime import datetime


from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from bike_rental_model import config
from bike_rental_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    dteday: Optional[datetime]  # Date of the record
    yr: Optional[int]  # Year (extracted from dteday)
    mnth: Optional[str]  # Month (extracted from dteday)
    season: Optional[str]  # Season (e.g., "winter")
    hr: Optional[str]  # Hour (e.g., "6am")
    holiday: Optional[str]  # Whether it's a holiday ("No" or "Yes")
    weekday: Optional[str]  # Day of the week (e.g., "Mon")
    workingday: Optional[str]  # Whether it's a working day ("No" or "Yes")
    weathersit: Optional[str]  # Weather situation (e.g., "Mist")
    temp: Optional[float]  # Normalized temperature in Celsius
    atemp: Optional[float]  # Normalized feeling temperature in Celsius
    hum: Optional[float]  # Humidity (0-100)
    windspeed: Optional[float]  # Windspeed (normalized)

   

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
