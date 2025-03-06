import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bike_rental_model import __version__ as _version
from bike_rental_model.config.core import config
from bike_rental_model.pipeline import bike_pipe
from bike_rental_model.processing.data_manager import load_pipeline
from bike_rental_model.processing.validations import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
bike_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = bike_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = bike_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":


    data_in = {
        'dteday': ['2012-05-13'],
        'season': ['summer'],
        'hr': ['12pm'],
        'holiday': ['No'],
        'weekday': ['Sun'],
        'workingday': ['No'],
        'weathersit': ['Clear'],
        'temp': [22.08],
        'atemp': [24.99],
        'hum': [56.99],
        'windspeed': [15.00],
        'casual': [189],
        'registered': [342]
    }

    # Now you can pass this into the DataFrame
    input_df = pd.DataFrame(data_in)
    make_prediction(input_data=input_df)