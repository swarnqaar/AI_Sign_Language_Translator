"""
Data Transformation Component
==============================
Responsible for:
  1. Separating features (126 landmark coords) from labels.
  2. Normalising the landmark coordinates with StandardScaler so that
     distance / scale differences between participants don't dominate.
  3. Imputing missing landmarks (hand not detected in a frame) with KNN.
  4. Persisting the fitted pipeline (imputer + scaler) as preprocessing.pkl
     so that inference at serving time uses the SAME transformations.
  5. Saving the transformed arrays as .npy files for the model trainer.

Why StandardScaler here instead of just KNNImputer?
  MediaPipe outputs (x, y, z) in normalised image coordinates [0, 1], but
  the z-axis has a different range. StandardScaler makes all 126 features
  zero-mean unit-variance, which helps distance-based classifiers (KNN,
  SVM) and speeds up gradient-based methods.
"""

import sys
import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from AI_signlanguage.constant.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)
from AI_signlanguage.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from AI_signlanguage.entity.config_entity import DataTransformationConfig
from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.logging.logger import logging
from AI_signlanguage.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise SignLanguageException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SignLanguageException(e, sys)

    # ------------------------------------------------------------------
    # Build the preprocessing pipeline
    # ------------------------------------------------------------------
    def get_data_transformer_object(self) -> Pipeline:
        """
        Returns a sklearn Pipeline with two steps:
          1. KNNImputer  – fills missing landmark coordinates using the
             5 nearest neighbours (weighted by inverse distance).
          2. StandardScaler – zero-mean, unit-variance normalisation.

        The pipeline is fitted only on the training split to prevent
        data leakage.
        """
        logging.info("Building data transformation pipeline.")
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            scaler = StandardScaler()
            processor = Pipeline([("imputer", imputer), ("scaler", scaler)])
            return processor
        except Exception as e:
            raise SignLanguageException(e, sys)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting data transformation.")
        try:
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Encode string labels → integers
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)
            y_train = label_encoder.transform(target_feature_train_df)
            y_test = label_encoder.transform(target_feature_test_df)

            # Fit pipeline on train, transform both splits
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_train = preprocessor_object.transform(input_feature_train_df)
            transformed_test = preprocessor_object.transform(input_feature_test_df)

            # Combine features + labels into single arrays
            train_arr = np.c_[transformed_train, y_train]
            test_arr = np.c_[transformed_test, y_test]

            # Persist transformed arrays
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )

            # Persist the fitted preprocessor
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object,
            )
            save_object("final_model/preprocessor.pkl", preprocessor_object)
            # Also save label encoder alongside preprocessor
            save_object("final_model/label_encoder.pkl", label_encoder)

            logging.info("Data transformation completed successfully.")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=(
                    self.data_transformation_config.transformed_object_file_path
                ),
                transformed_train_file_path=(
                    self.data_transformation_config.transformed_train_file_path
                ),
                transformed_test_file_path=(
                    self.data_transformation_config.transformed_test_file_path
                ),
            )
            return data_transformation_artifact
        except Exception as e:
            raise SignLanguageException(e, sys)
