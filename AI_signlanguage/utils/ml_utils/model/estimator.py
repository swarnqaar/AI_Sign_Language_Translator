import os
import sys
from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.logging.logger import logging
from AI_signlanguage.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME


class SignLanguageModel:
    """
    Wrapper combining the fitted preprocessor (StandardScaler pipeline)
    and the trained classifier into a single predict-able object.
    Mirrors NetworkModel from the reference project.
    """

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise SignLanguageException(e, sys)

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise SignLanguageException(e, sys)
