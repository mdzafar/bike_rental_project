from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class embarkImputer(BaseEstimator, TransformerMixin):
    """Embarked column Imputer"""

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

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
    """Categorical variable mapper."""

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
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X
    
class age_col_tfr(BaseEstimator, TransformerMixin):
	#""" Age column transformer"""

    def __init__(self, variables):
        
        if not isinstance(variables, str):
            raise ValueError('variables should be a str')
        
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
      self.age_avg = X[self.variables].mean()
      self.age_std = X[self.variables].std()
        # we need this step to fit the sklearn pipeline
      return self

    def transform(self, X):
        np.random.seed(42)
    	# so that we do not over-write the original dataframe
        X = X.copy()
        age_null_count = X[self.variables].isnull().sum()
        age_null_random_list = np.random.randint(self.age_avg - self.age_std, self.age_avg + self.age_std, size=age_null_count)
        X.loc[np.isnan(X[self.variables]),self.variables] = age_null_random_list
        X[self.variables] = X[self.variables].astype(int)

        return X
