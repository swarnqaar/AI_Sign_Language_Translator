"""
Data Validation Component
=========================
Responsible for:
  1. Checking that the ingested CSVs have the expected number of columns
     (as declared in data_schema/schema.yaml).
  2. Running a Kolmogorov-Smirnov drift test between train and test splits
     to detect distribution shift — important because gesture data can vary
     significantly across different camera setups or lighting conditions.
  3. Writing a YAML drift report for auditing.
  4. Returning a DataValidationArtifact that the next stage consumes.
"""

from AI_signlanguage.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from AI_signlanguage.entity.config_entity import DataValidationConfig
from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.logging.logger import logging
from AI_signlanguage.constant.training_pipeline import SCHEMA_FILE_PATH
from AI_signlanguage.utils.main_utils.utils import read_yaml_file, write_yaml_file

from scipy.stats import ks_2samp
import pandas as pd
import os
import sys


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SignLanguageException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            raise SignLanguageException(e, sys)

    # ------------------------------------------------------------------
    # Validation 1 – column count
    # ------------------------------------------------------------------
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Compares the actual column count in the DataFrame against the
        number of columns declared in schema.yaml.  A mismatch usually
        means a data collection pipeline change broke the feature set.
        """
        try:
            number_of_columns = len(self._schema_config.get("columns", []))
            logging.info(f"Required columns: {number_of_columns}")
            logging.info(f"DataFrame columns: {len(dataframe.columns)}")
            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise SignLanguageException(e, sys)

    # ------------------------------------------------------------------
    # Validation 2 – distribution drift (KS test)
    # ------------------------------------------------------------------
    def detect_dataset_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        try:
            status = True
            report = {}

            numeric_cols = [
                c for c in base_df.columns
                if c != "label" and pd.api.types.is_numeric_dtype(base_df[c])
            ]

            for column in numeric_cols:
                # Drop NaN values before KS test
                d1 = base_df[column].dropna().astype(float).values
                d2 = current_df[column].dropna().astype(float).values

                # Skip column if not enough data
                if len(d1) < 2 or len(d2) < 2:
                    continue

                try:
                    stat, p_value = ks_2samp(d1, d2)
                    drift_found = bool(p_value < threshold)
                    if drift_found:
                        status = False
                    report[column] = {
                        "p_value": float(p_value),
                        "drift_status": drift_found,
                    }
                except Exception:
                    continue

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            logging.info(f"Drift report saved. Status: {status}")
            return status

        except Exception as e:
            raise SignLanguageException(e, sys)
    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Column count checks
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                logging.warning("Train DataFrame does not contain all schema columns.")

            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                logging.warning("Test DataFrame does not contain all schema columns.")

            # Drift detection
            status = self.detect_dataset_drift(
                base_df=train_dataframe, current_df=test_dataframe
            )

            # Save validated files
            dir_path = os.path.dirname(
                self.data_validation_config.valid_train_file_path
            )
            os.makedirs(dir_path, exist_ok=True)
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False,
                header=True,
            )
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False,
                header=True,
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise SignLanguageException(e, sys)
