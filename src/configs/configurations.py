import os
from dataclasses import dataclass

ARTIFACT_FOLDER_PATH = "artifacts"
NOTEBOOK_DATA_FOLDER_PATH = "notebooks/data"
HAMLET_TEXT_FILE_NAME = "hamlet.txt"
TOKENIZER_FILE_NAME = "tokenizer.pkl"
PREPROCESSED_DATASET_FILE_NAME = "preprocessed_data.csv"
TRAINED_MODEL_KERAS_FILE_NAME = "model.keras"


@dataclass
class DataIngestionConfig:
    hamlet_text_file_path: str = os.path.join(NOTEBOOK_DATA_FOLDER_PATH, HAMLET_TEXT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    tokenizer_path: str = os.path.join(ARTIFACT_FOLDER_PATH, TOKENIZER_FILE_NAME)
    preprocessed_data_path: str = os.path.join(ARTIFACT_FOLDER_PATH, PREPROCESSED_DATASET_FILE_NAME)

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join(ARTIFACT_FOLDER_PATH, TRAINED_MODEL_KERAS_FILE_NAME)

@dataclass
class PredictPipelineConfig:
    model_path: str = os.path.join(ARTIFACT_FOLDER_PATH, TRAINED_MODEL_KERAS_FILE_NAME)
    tokenizer_path: str = os.path.join(ARTIFACT_FOLDER_PATH, TOKENIZER_FILE_NAME)
