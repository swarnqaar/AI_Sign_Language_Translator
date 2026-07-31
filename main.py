"""
main.py  –  Trigger the full training pipeline from the CLI
============================================================
Usage:
    python main.py
"""

from AI_signlanguage.components.data_ingestion import DataIngestion
from AI_signlanguage.components.data_validation import DataValidation
from AI_signlanguage.components.data_transformation import DataTransformation
from AI_signlanguage.components.model_trainer import ModelTrainer
from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.logging.logger import logging
from AI_signlanguage.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
import sys

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()

        # Step 1 – Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        print(data_ingestion_artifact)

        # Step 2 – Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(
            data_ingestion_artifact, data_validation_config
        )
        logging.info("Initiating data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)

        # Step 3 – Data Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        logging.info("Initiating data transformation")
        data_transformation = DataTransformation(
            data_validation_artifact, data_transformation_config
        )
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )
        print(data_transformation_artifact)
        logging.info("Data transformation completed")

        # Step 4 – Model Training
        logging.info("Initiating model training")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact,
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed")
        print(model_trainer_artifact)

    except Exception as e:
        raise SignLanguageException(e, sys)
