import os
import numpy as np

"""
Defining common constant variables for training pipeline
"""
TARGET_COLUMN = "label"
PIPELINE_NAME: str = "SignLanguageTranslator"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "sign_language_data.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"

# Data Ingestion
DATA_INGESTION_COLLECTION_NAME: str = "SignLanguageData"
DATA_INGESTION_DATABASE_NAME: str = "SIGNLANGDB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# Data Validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"


# Data Transformation
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 5,
    "weights": "distance",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"




# Sign language gesture class labels (ASL A-Z + common phrases)
SIGN_CLASSES = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "Hello","Thank You","Yes","No","Please","Sorry",
    "Help","More","Done","Good",
]
NUM_CLASSES = len(SIGN_CLASSES)

# MediaPipe: 21 landmarks x 3 coords (x,y,z) x 2 hands = 126 features
NUM_LANDMARKS = 21
NUM_COORDS_PER_LANDMARK = 3
NUM_HANDS = 2
FEATURE_SIZE = NUM_LANDMARKS * NUM_COORDS_PER_LANDMARK * NUM_HANDS  # 126
