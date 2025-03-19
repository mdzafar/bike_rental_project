import pandas as pd
from bike_rental_model.config.core import config
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 
from bike_rental_model.processing.features import WeekdayImputer, WeathersitImputer, Mapper, OutlierHandler, WeekdayOneHotEncoder


# ------------------------- Test WeekdayImputer -------------------------
def test_weekday_imputer(sample_input_data):
    """Test imputation of missing values in the 'weekday' column"""
    
    X_test, _ = sample_input_data  # We only need X_test for transformations
     # Ensure 'dteday' is in datetime format
    X_test['dteday'] = pd.to_datetime(X_test['dteday'], errors='coerce')
    # Introduce missing values for testing
    X_test.loc[X_test.index[0], 'weekday'] = None
    X_test.loc[X_test.index[3], 'weekday'] = None

    imputer = WeekdayImputer(date_column='dteday', weekday_column='weekday')
    transformed_data = imputer.transform(X_test)
    print(list(transformed_data['weekday']))
    # Ensure 'dteday' column is dropped
    assert 'dteday' not in transformed_data.columns, "dteday column should be dropped"

    # Validate weekday imputation (Recompute expected weekday names)
    expected_weekdays = X_test['dteday'].dt.day_name().str[:3].tolist()
    print(expected_weekdays)
    assert list(transformed_data['weekday']) == expected_weekdays, "Weekday imputation failed"

# ------------------------- Test WeathersitImputer -------------------------
def test_weathersit_imputer(sample_input_data):
    """Test imputation of missing values in the 'weathersit' column"""

    X_test, _ = sample_input_data  # Only need X_test

    # Introduce missing values
    X_test.loc[X_test.index[2], 'weathersit'] = None
    X_test.loc[X_test.index[4], 'weathersit'] = None

    imputer = WeathersitImputer(variables='weathersit')
    imputer.fit(X_test)
    transformed_data = imputer.transform(X_test)

    most_frequent = X_test['weathersit'].mode()[0]  # Most frequent value

    assert transformed_data['weathersit'].isnull().sum() == 0, "Null values should be imputed"
    assert all(transformed_data['weathersit'] == transformed_data['weathersit'].fillna(most_frequent)), "Incorrect imputation"


def get_year_and_month(dataframe):

    df = dataframe.copy()
    # convert 'dteday' column to Datetime datatype
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    df['yr'] = df['dteday'].dt.year
    df['mnth'] = df['dteday'].dt.month_name()

    return df

def test_get_year_and_month(sample_input_data):
    """Test if the function correctly extracts year and month from dteday"""
    X_test, _ = sample_input_data  # Only need X_test
    transformed_data = get_year_and_month(X_test)

    # Ensure new columns exist
    assert 'yr' in transformed_data.columns, "'yr' column is missing"
    assert 'mnth' in transformed_data.columns, "'mnth' column is missing"

    # Ensure 'yr' contains only 2011 or 2012
    assert set(transformed_data['yr'].unique()) <= {2011, 2012}, "Unexpected year values found"

    # Ensure 'mnth' contains valid month names
    expected_months = set([
        'January', 'February', 'March', 'April', 'May', 'June', 'July', 
        'August', 'September', 'October', 'November', 'December'
    ])
    assert set(transformed_data['mnth'].unique()).issubset(expected_months), "Unexpected month names found"

    # Check expected values
    expected_years = [2012, 2011]
    expected_months = ['April', 'June', 'December', 'July', 'February', 'November', 'October', 'March', 'August', 'January', 'September', 'May']
    print((transformed_data['yr'].unique()).tolist())
    print(list(transformed_data['mnth'].unique()))
    assert list(transformed_data['yr'].unique()) == expected_years, "Year column values are incorrect"
    assert list(transformed_data['mnth'].unique()) == expected_months, "Month column values are incorrect"

# ------------------------- Test Mapper -------------------------
def test_mapper(sample_input_data):
    """Test mapping categorical variables to numerical values"""

    X_test, _ = sample_input_data  # Only need X_test
    
    # Ensure there are no unexpected missing values
    X_test = X_test.copy()  # Avoid modifying the original test dataset

    X_test =get_year_and_month(X_test)
    X_test['weathersit'] = X_test['weathersit'].fillna('Clear')  # Assign a default value

    mapping_dict = {'Heavy Rain': 0, 'Light Rain': 1, 'Mist': 2, 'Clear': 3}
    map_weathersit = Mapper(variables='weathersit', mappings=mapping_dict)
    transformed_data = map_weathersit.transform(X_test)

    # Ensure no NaN values after mapping
    assert transformed_data['weathersit'].isna().sum() == 0, "NaN values found after mapping"
    # Ensure only expected values are present
    assert all(transformed_data['weathersit'].isin(mapping_dict.values())), "Mapping did not apply correctly"

    mapping_dict =  {'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3} 
    map_season=Mapper('season', {'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3} )
    transformed_data = map_season.transform(X_test)
    assert transformed_data['season'].isna().sum() == 0, "NaN values found after mapping"
    assert all(transformed_data['season'].isin(mapping_dict.values())), "Mapping did not apply correctly"

    mapping_dict ={'No': 0, 'Yes': 1}
    map_workingday = Mapper('workingday',{'No': 0, 'Yes': 1})
    transformed_data = map_workingday.transform(X_test)
    assert transformed_data['workingday'].isna().sum() == 0, "NaN values found after mapping"
    assert all(transformed_data['workingday'].isin(mapping_dict.values())), "Mapping did not apply correctly"

    
    mapping_dict ={'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, '12am': 5, '6am': 6, '11pm': 7, '10pm': 8,
                '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, '8pm': 14, '2pm': 15, '1pm': 16,
                '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, '6pm': 22, '5pm': 23}
    map_hr =  Mapper('hr',{'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, '12am': 5, '6am': 6, '11pm': 7, '10pm': 8,
                '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, '8pm': 14, '2pm': 15, '1pm': 16,
                '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, '6pm': 22, '5pm': 23})
    transformed_data = map_hr.transform(X_test)
    assert transformed_data['hr'].isna().sum() == 0, "NaN values found after mapping"
    #map_hr.fit(transformed_data).transform(transformed_data)
    assert all(transformed_data['hr'].isin(mapping_dict.values())), "Mapping did not apply correctly"
    
    
    mapping_dict ={'Yes': 0, 'No': 1}
    map_holiday = Mapper('holiday',{'Yes': 0, 'No': 1})
    assert sample_input_data[0]['holiday'].isin(['Yes','No']).sum() > 0
    transformed_data = map_holiday.transform(X_test)
    assert transformed_data['holiday'].isin(['Yes','No']).sum() == 0
    assert transformed_data['holiday'].isin([1,0]).sum() > 0
    
    mapping_dict ={2011: 0, 2012: 1}
    map_yr = Mapper('yr',{2011: 0, 2012: 1})
    transformed_data = map_yr.transform(X_test)
    assert transformed_data['yr'].isna().sum() == 0, "NaN values found after mapping"
    assert all(transformed_data['yr'].isin(mapping_dict.values())), "Mapping did not apply correctly"

    mapping_dict ={'January': 0, 'February': 1, 'December': 2, 'March': 3, 'November': 4, 'April': 5,
                'October': 6, 'May': 7, 'September': 8, 'June': 9, 'July': 10, 'August': 11}
    map_mnth = Mapper('mnth' ,{'January': 0, 'February': 1, 'December': 2, 'March': 3, 'November': 4, 'April': 5,
                'October': 6, 'May': 7, 'September': 8, 'June': 9, 'July': 10, 'August': 11})
    transformed_data = map_mnth.transform(X_test)
    assert transformed_data['mnth'].isna().sum() == 0, "NaN values found after mapping"
    assert all(transformed_data['mnth'].isin(mapping_dict.values())), "Mapping did not apply correctly"


# ------------------------- Test OutlierHandler -------------------------
def test_outlier_handler(sample_input_data):
    """Test handling of outliers in numerical features"""

    X_test, _ = sample_input_data  # Only need X_test

    handler = OutlierHandler(variable='temp')
    handler.fit(X_test)
    transformed_data = handler.transform(X_test)

    lower_bound, upper_bound = handler.lower_bound, handler.upper_bound

    # Ensure all values are within the bounds
    assert all(transformed_data['temp'] >= lower_bound), "Some values are below lower bound"
    assert all(transformed_data['temp'] <= upper_bound), "Some values are above upper bound"

# ------------------------- Test WeekdayOneHotEncoder -------------------------
def test_weekday_onehot_encoder(sample_input_data):
    """Test one-hot encoding of the 'weekday' column"""

    X_test, _ = sample_input_data  # Only need X_test
    X_test = X_test.copy()
    encoder = WeekdayOneHotEncoder(column='weekday')
    encoder.fit(X_test)
    transformed_data = encoder.transform(X_test)

    expected_columns = encoder.enc_wkday_features  # One-hot encoded feature names

    # Ensure one-hot encoded columns exist
    for col in expected_columns:
        assert col in transformed_data.columns, f"Column {col} missing in transformed data"

    # Ensure original 'weekday' column is dropped
    assert 'weekday' not in transformed_data.columns, "Original column should be dropped"
