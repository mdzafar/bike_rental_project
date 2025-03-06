
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, date_column, weekday_column = 'weekday'):

        self.date_column = date_column
        self.weekday_column = weekday_column

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      X = X.copy()
      weeks=X[X[self.weekday_column].isnull()].index
      X.loc[weeks,self.weekday_column] = X.loc[weeks,self.date_column].dt.day_name().str[:3]
      X.drop(self.date_column, axis=1, inplace=True)
      return X


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        self.fill_value=X[self.variables].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables]=X[self.variables].fillna(self.fill_value)

        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for feature in self.variables:
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable:str):
      if not isinstance(variable, str):
          raise ValueError("variables should be a string")
      self.variable = variable


    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
      q1 = X.describe()[self.variable].loc['25%']
      q3 = X.describe()[self.variable].loc['75%']
      iqr = q3 - q1
      self.lower_bound = q1 - (1.5 * iqr)
      self.upper_bound = q3 + (1.5 * iqr)
      return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      X = X.copy()
      #for col in X.columns:
      # q1 = X.describe()[col].loc['25%']
      # q3 = X.describe()[col].loc['75%']
      # iqr = q3 - q1
      # lower_bound = q1 - (1.5 * iqr)
      # upper_bound = q3 + (1.5 * iqr)
      for i in X.index:
        if X.loc[i,self.variable] > self.upper_bound:
          X.loc[i,self.variable]= self.upper_bound
        if X.loc[i,self.variable] < self.lower_bound:
          X.loc[i,self.variable]= self.lower_bound
      return X



class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, column):
        self.encoder = OneHotEncoder(sparse_output=False)
        self.column = column

    def fit(self, X, y = None):
      X=X.copy()
      self.encoder.fit(X[[self.column]])
      self.enc_wkday_features = self.encoder.get_feature_names_out([self.column])
      return self

    def transform(self, X):
      X=X.copy()
      encoder_df=self.encoder.transform(X[[self.column]])
      X[self.enc_wkday_features] = encoder_df
        # drop 'weekday' column after encoding
      X.drop(self.column, axis=1, inplace=True)

      return X
