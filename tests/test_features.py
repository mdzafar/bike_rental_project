import pytest
import pandas as pd
import numpy as np
from bike_rental_model.config.core import config
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 
from bike_rental_model.processing.features import WeekdayImputer, WeathersitImputer, Mapper, OutlierHandler, WeekdayOneHotEncoder

@pytest.fixture
def pipeline():
    return Pipeline([
        ('weekday_imputer', WeekdayImputer(date_column='dteday')),
        ('weekday_encoder', WeekdayOneHotEncoder(column='weekday')),
        ('weathersit_imputer', WeathersitImputer(variables='weathersit')),
        ('outlier_handler', OutlierHandler(variable='temp')),
        ('scaler', StandardScaler())    
    ])

# def test_weekday_imputer(sample_input_data):
#     # imputer = WeekdayImputer(date_column='dteday')
#     # transformed = imputer.fit_transform(sample_input_data)
#     # assert transformed['weekday'].isnull().sum() == 0
#     transformer = WeekdayImputer(
#         config.model_config_.dteday_var.dt.strftime('%Y-%m-%d'),
#         config.model_config_.weekday_var # cabin
#     )
    
#     print("before",set(sample_input_data[0]["weekday"].to_list()))
#     subject = transformer.transform(sample_input_data[0])
#     # assert np.isnan(sample_input_data[0].iloc[7,4])
#     print("after",set(sample_input_data[0]["weekday"].to_list()))
#     assert True == False

def test_weathersit_imputer(sample_input_data):
    transformer = WeathersitImputer(
        config.model_config_.weathersit_var
    )
    assert sample_input_data[0]["weathersit"].isnull().sum() > 0
    # assert np.isnan(sample_input_data[0]["weathersit"]).sum() > 0
    # assert True == False
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])
    assert subject["weathersit"].isnull().sum() == 0

# def test_outlier_handler(sample_data):
#     handler = OutlierHandler(variable='temp')
#     transformed = handler.fit_transform(sample_data)
#     assert transformed['temp'].max() <= handler.upper_bound
#     assert transformed['temp'].min() >= handler.lower_bound

# def test_pipeline_transforms(sample_data, pipeline):
#     transformed_data = pipeline.fit_transform(sample_data)
#     assert transformed_data is not None

# def test_prediction_steps(sample_data):
#     X = sample_data[['temp', 'windspeed']]
#     y = sample_data['cnt']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     model = RandomForestRegressor()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     assert mean_squared_error(y_test, y_pred) >= 0
#     assert r2_score(y_test, y_pred) <= 1