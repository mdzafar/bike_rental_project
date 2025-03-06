import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from titanic_model.config.core import config
from titanic_model.processing.features import embarkImputer
from titanic_model.processing.features import Mapper
from titanic_model.processing.features import age_col_tfr

titanic_pipe=Pipeline([
    
    ("embark_imputation", embarkImputer(variables=config.model_config_.embarked_var)
     ),
     ##==========Mapper======##
     ("map_sex", Mapper(config.model_config_.gender_var, config.model_config_.gender_mappings)
      ),
     ("map_embarked", Mapper(config.model_config_.embarked_var, config.model_config_.embarked_mappings )
     ),
     ("map_title", Mapper(config.model_config_.title_var, config.model_config_.title_mappings)
     ),
     # Transformation of age column
     ("age_transform", age_col_tfr(config.model_config_.age_var)
     ),
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestClassifier(n_estimators=config.model_config_.n_estimators, 
                                         max_depth=config.model_config_.max_depth, 
                                         max_features=config.model_config_.max_features,
                                         random_state=config.model_config_.random_state))
          
     ])
