import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from bike_rental_model.config.core import config
from bike_rental_model.pipeline import bike_pipe
from bike_rental_model.processing.data_manager import load_dataset, save_pipeline
from sklearn.metrics import mean_squared_error, r2_score


def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config_.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config_.features],  # predictors
        data[config.model_config_.target],
        test_size=config.model_config_.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config_.random_state,
    )

    # Pipeline fitting
    bike_pipe.fit(X_train,y_train)
    y_pred = bike_pipe.predict(X_test)
    #print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist= bike_pipe)
    
    print("Accuracy(in %):", r2_score(y_test, y_pred)*100)

    # printing the score
    
if __name__ == "__main__":
    run_training()
