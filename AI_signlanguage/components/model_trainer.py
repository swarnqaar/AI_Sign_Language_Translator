"""
Model Trainer Component
=======================
Responsible for:
  1. Loading the transformed .npy arrays.
  2. Running GridSearchCV across five classifiers to find the best model.
  3. Logging metrics (F1, precision, recall, accuracy) to MLflow/DagsHub.
  4. Saving the winning model as final_model/model.pkl.
  5. Returning a ModelTrainerArtifact.

Classifier selection rationale for sign-language gesture recognition
(126 landmark features, 36 classes):
  - Random Forest / Gradient Boosting:  strong on tabular landmark data,
    naturally multi-class.
  - SVC:  historically excellent on skeleton/pose feature vectors.
  - KNN:  simple baseline that often works surprisingly well on
    normalised landmark coords.
  - Logistic Regression:  linear baseline; interpretable.
"""

import os
import sys

from AI_signlanguage.entity.artifact_entity import ModelTrainerArtifact
from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.entity.artifact_entity import ModelTrainerArtifact
from AI_signlanguage.logging.logger import logging

from AI_signlanguage.entity.artifact_entity import ModelTrainerArtifact
from AI_signlanguage.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from AI_signlanguage.entity.artifact_entity import ModelTrainerArtifact
from AI_signlanguage.entity.config_entity import ModelTrainerConfig
from AI_signlanguage.entity.artifact_entity import ModelTrainerArtifact
from AI_signlanguage.utils.ml_utils.model.estimator import SignLanguageModel
from AI_signlanguage.entity.artifact_entity import ModelTrainerArtifact
from AI_signlanguage.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from AI_signlanguage.entity.artifact_entity import ModelTrainerArtifact
from AI_signlanguage.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)



class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SignLanguageException(e, sys)

   


    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------
    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Defines candidate classifiers + their hyper-parameter grids,
        calls evaluate_models() which runs GridSearchCV internally,
        then selects the best model by test-set accuracy.
        """
        models = {
            "Random Forest": RandomForestClassifier(n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVC": SVC(probability=True),
            "KNN": KNeighborsClassifier(n_jobs=-1),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, multi_class="auto", n_jobs=-1
            ),
        }

        params = {
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
            },
            "Gradient Boosting": {
                "learning_rate": [0.05, 0.1, 0.2],
                "n_estimators": [100, 200],
                "subsample": [0.8, 1.0],
            },
            "SVC": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
            },
            "KNN": {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"],
            },
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10],
            },
        }

        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        # Pick best
        best_model_score = max(model_report.values())
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        logging.info(
            f"Best model: {best_model_name} | Test accuracy: {best_model_score:.4f}"
        )

        # Compute detailed metrics and log to MLflow
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_train, y_train_pred)
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_test, y_test_pred)
        self.track_mlflow(best_model, classification_test_metric)

        # Check for over/under-fitting
        gap = abs(
            classification_train_metric.accuracy - classification_test_metric.accuracy
        )
        if gap > self.model_trainer_config.overfitting_underfitting_threshold:
            logging.warning(
                f"Large train/test accuracy gap ({gap:.4f}) — possible overfitting."
            )

        if (
            classification_test_metric.accuracy
            < self.model_trainer_config.expected_accuracy
        ):
            raise Exception(
                f"Model accuracy {classification_test_metric.accuracy:.4f} is below "
                f"the expected threshold {self.model_trainer_config.expected_accuracy}"
            )

        # Persist
        preprocessor = load_object(
            self.data_transformation_artifact.transformed_object_file_path
        )
        model_dir_path = os.path.dirname(
            self.model_trainer_config.trained_model_file_path
        )
        os.makedirs(model_dir_path, exist_ok=True)

        sign_model = SignLanguageModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=sign_model)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact
        except Exception as e:
            raise SignLanguageException(e, sys)
