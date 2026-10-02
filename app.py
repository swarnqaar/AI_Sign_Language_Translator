
"""
app.py  –  FastAPI Application
================================
Endpoints:
  GET  /          → redirect to /docs
  GET  /train     → trigger the full training pipeline
  POST /predict   → upload a CSV of landmark features → returns prediction table
  POST /live      → accept a JSON payload of one frame's landmark vector → returns gesture label

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URL")

import pymongo
from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.logging.logger import logging
from AI_signlanguage.constant import training_pipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
import pandas as pd
import numpy as np

from AI_signlanguage.utils.main_utils.utils import load_object
from AI_signlanguage.utils.ml_utils.model.estimator import SignLanguageModel
from AI_signlanguage.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# MongoDB client
try:
    client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca, serverSelectionTimeoutMS=8000)
    client.server_info()  # test connection
    print("MongoDB connected OK")
except Exception as e:
    print(f"MongoDB not connected: {e}")
    client = None

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI(
    title="AI Sign Language Translator",
    description=(
        "Translate American Sign Language (ASL) gestures to text "
        "using MediaPipe hand landmarks + a trained classifier."
    ),
    version="1.0.0",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["Training"])
async def train_route():
    """Trigger the full ML training pipeline (ingestion → validation → transformation → training)."""
    try:
        train_pipeline = training_pipeline.TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training pipeline completed successfully.")
    except Exception as e:
        raise SignLanguageException(e, sys)


@app.post("/predict", tags=["Inference"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    """
    Upload a CSV file of MediaPipe landmark features (one row per frame).
    Returns an HTML table with a 'predicted_gesture' column appended.
    """
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        label_encoder = load_object("final_model/label_encoder.pkl")

        sign_model = SignLanguageModel(preprocessor=preprocessor, model=final_model)
        y_pred_encoded = sign_model.predict(df)
        y_pred_labels = label_encoder.inverse_transform(y_pred_encoded.astype(int))

        df["predicted_gesture"] = y_pred_labels

        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped table-hover")
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html}
        )
    except Exception as e:
        raise SignLanguageException(e, sys)


from pydantic import BaseModel
from typing import List

class LandmarkInput(BaseModel):
    landmarks: List[float]

@app.post("/live", tags=["Inference"])
async def live_predict(data: LandmarkInput):
    """
    Send 126 landmark float values and get back a gesture label.
    
    Example body:
    {
        "landmarks": [0.12, 0.45, 0.03, ...]
    }
    """
    try:
        landmarks = data.landmarks

        if len(landmarks) != 126:
            return JSONResponse(
                {"error": f"Expected 126 values, got {len(landmarks)}"},
                status_code=400,
            )

        # Check model files exist
        preprocessor_path  = "final_model/preprocessor.pkl"
        model_path         = "final_model/model.pkl"
        label_encoder_path = "final_model/label_encoder.pkl"

        if not os.path.exists(model_path):
            return JSONResponse(
                {"error": "Model not found. Train the model first."},
                status_code=404,
            )

        preprocessor  = load_object(preprocessor_path)
        final_model   = load_object(model_path)
        label_encoder = load_object(label_encoder_path)

        sign_model     = SignLanguageModel(preprocessor=preprocessor, model=final_model)
        feature_vector = np.array(landmarks).reshape(1, -1)
        y_pred_encoded = sign_model.model.predict(
            preprocessor.transform(feature_vector)
        )
        gesture = label_encoder.inverse_transform(
            y_pred_encoded.astype(int)
        )[0]

        return JSONResponse({"gesture": gesture, "status": "success"})

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )

'''
@app.post("/live", tags=["Inference"])
async def live_predict(request: Request):
    """
    Accept a JSON body with a single 'landmarks' key containing a list of
    126 float values (21 landmarks × 3 coords × 2 hands, left hand first).
    Returns the predicted gesture label.

    Example body:
        {"landmarks": [0.12, 0.45, 0.03, ...]}   # 126 floats
    """
    try:
        body = await request.json()
        landmarks = body.get("landmarks", [])

        if len(landmarks) != 126:
            return JSONResponse(
                {"error": f"Expected 126 landmark values, got {len(landmarks)}."},
                status_code=400,
            )

        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        label_encoder = load_object("final_model/label_encoder.pkl")

        sign_model = SignLanguageModel(preprocessor=preprocessor, model=final_model)
        feature_vector = np.array(landmarks).reshape(1, -1)
        y_pred_encoded = sign_model.model.predict(
            preprocessor.transform(feature_vector)
        )
        gesture = label_encoder.inverse_transform(y_pred_encoded.astype(int))[0]

        return JSONResponse({"gesture": gesture})
    except Exception as e:
        raise SignLanguageException(e, sys)
# exception handling for the entire app

'''

if __name__ == "__main__":
    app_run(app, host="127.0.0.1", port=8000)
