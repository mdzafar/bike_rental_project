from sklearn.pipeline import Pipeline
from bike_rental_model.processing.features import WeekdayImputer,WeathersitImputer, Mapper, OutlierHandler, WeekdayOneHotEncoder
from bike_rental_model.config.core import config
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

bike_pipe = Pipeline([

    ('WeekdayImputation', WeekdayImputer(config.model_config_.dteday_var)),
    ('weathersit imputation', WeathersitImputer(config.model_config_.weathersit_var)),

    ##==========Mapper======##
    ('map_yr', Mapper(config.model_config_.yr_var,config.model_config_.year_mappings)),
    ('map_mnth', Mapper(config.model_config_.month_var ,config.model_config_.month_mappings)),
    ('map_season', Mapper(config.model_config_.season_var,config.model_config_.season_mappings)),
    ('map_weathersit', Mapper(config.model_config_.weathersit_var,config.model_config_.weather_mappings)),
    ('map_holiday' , Mapper(config.model_config_.holiday_var,config.model_config_.holiday_mappings)),
    ('map_workingday' , Mapper(config.model_config_.workingday_var,config.model_config_.workingday_mappings)),
    ('map_hr' , Mapper(config.model_config_.hr_var,config.model_config_.hour_mappings)),
  

    # outlier
    ('outlier handler1', OutlierHandler(config.model_config_.temp_var)),
    ('outlier handler2', OutlierHandler(config.model_config_.atemp_var)),
    ('outlier handler3', OutlierHandler( config.model_config_.hum_var)),
    ('outlier handler4', OutlierHandler(config.model_config_.wind_speed_var)),
    #onehotnecoding
    ('one hot encoding', WeekdayOneHotEncoder(config.model_config_.weekday_var)),

    # scale 
    ('scaler', StandardScaler()),

    # Model fit
    ('model_rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])