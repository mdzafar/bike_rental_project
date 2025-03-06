# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
sys.path.append(str(file.parents[2]))
# sys.path.append(str(file.parents[3]))
print(file.parents[1])
# print(sys.path)
from typing import Dict, List
from pydantic import BaseModel
from strictyaml import YAML, load

import bike_rental_model

# Project Directories
PACKAGE_ROOT = Path(bike_rental_model.__file__).resolve().parent
#print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    training_data_file: str
    pipeline_save_file: str
    pipeline_name: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    target: str
    features: List[str]
    unused_fields: List[str]
    dteday_var: str 
    weathersit_var:str 
    temp_var:str 
    atemp_var: str 
    hum_var:str
    yr_var:str
    month_var: str
    season_var: str
    hr_var: str
    holiday_var: str
    workingday_var: str
    wind_speed_var:str
    weekday_var:str
    year_mappings: Dict[int, int] 
    month_mappings: Dict[str, int]
    season_mappings: Dict[str, int]
    weather_mappings: Dict[str, int]
    holiday_mappings: Dict[str, int]
    workingday_mappings: Dict[str, int]
    hour_mappings: Dict[str, int]
    test_size:float
    random_state: int
    n_estimators: int
    max_depth: int
    max_features: int



class Config(BaseModel):
    """Master config object."""

    app_config_: AppConfig
    model_config_: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config_=AppConfig(**parsed_config.data),
        model_config_=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()