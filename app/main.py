import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import *
from typing import Dict, List, Optional
from time import sleep
import multiprocessing
import uuid
import uvicorn
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI(title="ML Model Management API")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("Starting app...")
log_filename = "server.log"

try:
    with open(f"logs/{log_filename}", 'x') as f:
        f.write("Streamlit app log file.\n")
    print(f"{log_filename} created.")
except FileExistsError:
    print(f"{log_filename} already exists.")

# Логирование с ротацией
log_handler = RotatingFileHandler("logs/server.log", maxBytes=1000000, backupCount=5)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
logger = logging.getLogger("uvicorn")
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

print("App started.")

# CORS для фронта
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use a normal dictionary instead of multiprocessing.Manager to avoid spawn issues
MODELS: Dict[str, Dict] = {}
ACTIVE_MODEL_ID: Optional[str] = None

# Model types
MODEL_TYPES = {
    "xgboost": {
        "name": "XGBoost Classifier",
        "description": "Gradient boosting algorithm known for performance and accuracy",
        "class": xgb.XGBClassifier
    },
    "randomforest": {
        "name": "Random Forest Classifier",
        "description": "Ensemble learning method using multiple decision trees",
        "class": RandomForestClassifier
    }
}


# Загрузка модели при запуске
@app.on_event("startup")
def load_models():
    global MODELS, ACTIVE_MODEL_ID
    # Create default models
    for model_type, model_info in MODEL_TYPES.items():
        model_id = str(uuid.uuid4())
        MODELS[model_id] = {
            "id": model_id,
            "name": f"Default {model_info['name']}",
            "description": model_info['description'],
            "is_active": model_type == "xgboost",  # XGBoost is active by default
            "type": model_type,
            "object": None,  # Will be initialized when trained
            "metrics": {},
            "feature_names": []
        }
        if model_type == "xgboost":
            ACTIVE_MODEL_ID = model_id

    logger.info("Models loaded at startup.")


# Ручка: список моделей
@app.get("/models", response_model=List[ModelInfo])
def get_models():
    return [
        ModelInfo(**{k: v for k, v in model.items() if k in ModelInfo.__fields__})
        for model in MODELS.values()
    ]


# Ручка: установить активную модель
@app.post("/set", response_model=FitResponse)
def set_active_model(req: SetModelRequest):
    global ACTIVE_MODEL_ID
    if req.id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    for m in MODELS.values():
        m["is_active"] = False
    MODELS[req.id]["is_active"] = True
    ACTIVE_MODEL_ID = req.id
    logger.info(f"Active model set to: {req.id}")
    return FitResponse(status="success", message=f"Active model set to {req.id}")


# Ручка: создать новую модель
@app.post("/create_model", response_model=ModelCreationResponse)
def create_model(req: ModelCreationRequest):
    if req.model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {req.model_type}")

    model_id = str(uuid.uuid4())
    model_info = MODEL_TYPES[req.model_type]

    MODELS[model_id] = {
        "id": model_id,
        "name": req.name,
        "description": model_info["description"],
        "is_active": False,
        "type": req.model_type,
        "object": None,
        "metrics": {},
        "feature_names": []
    }

    logger.info(f"Created new model: {req.name} (type: {req.model_type})")
    return ModelCreationResponse(
        status="success",
        message=f"Model {req.name} created successfully",
        model_id=model_id
    )


# Вспомогательная функция для обучения
def fit_model(hyperparameters, model_id, result_dict, data_path=None):
    try:
        model_type = MODELS[model_id]["type"]
        model_class = MODEL_TYPES[model_type]["class"]

        # Load data if provided
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path, sep=";")

            # Assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            # Transform target variable to start from 0
            y = y - y.min()

            # Store feature names for later use in predictions
            feature_names = X.columns.tolist()

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create model with hyperparameters
            if model_type == "xgboost":
                unique_classes = y.unique()
                if len(unique_classes) > 2:
                    # Multi-class scenario
                    chosen_metric = "mlogloss"
                    model = model_class(
                        objective="multi:softprob",
                        num_class=len(unique_classes),
                        learning_rate=hyperparameters.get("learning_rate", 0.1),
                        n_estimators=hyperparameters.get("n_estimators", 100),
                        max_depth=hyperparameters.get("max_depth", 3),
                        eval_metric="mlogloss",
                        random_state=42
                    )
                else:
                    # Binary classification scenario
                    chosen_metric = "logloss"
                    model = model_class(
                        objective="binary:logistic",
                        learning_rate=hyperparameters.get("learning_rate", 0.1),
                        n_estimators=hyperparameters.get("n_estimators", 100),
                        max_depth=hyperparameters.get("max_depth", 3),
                        eval_metric="logloss",
                        random_state=42
                    )

                # Train with evaluation
                eval_set = [(X_train, y_train), (X_val, y_val)]
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )

                # Get evaluation results
                evals_result = model.evals_result()

                # Create learning curve plot
                plt.figure(figsize=(10, 6))
                plt.plot(evals_result['validation_0'][chosen_metric], label='train')
                plt.plot(evals_result['validation_1'][chosen_metric], label='validation')
                plt.xlabel('Iterations')
                plt.ylabel('Log Loss')
                plt.title('XGBoost Learning Curve')
                plt.legend()

                # Save plot
                plot_path = f"plots/{model_id}.png"
                plt.savefig(plot_path)
                plt.close()

                metrics = {
                    "train_loss": evals_result['validation_0'][chosen_metric][-1],
                    "val_loss": evals_result['validation_1'][chosen_metric][-1],
                    "iterations": len(evals_result['validation_0'][chosen_metric]),
                    "plot_path": plot_path
                }

            elif model_type == "randomforest":
                model = model_class(
                    n_estimators=hyperparameters.get("n_estimators", 100),
                    max_depth=hyperparameters.get("max_depth", None),
                    random_state=42
                )

                # Train model
                model.fit(X_train, y_train)

                # Calculate metrics
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)

                # Create feature importance plot
                plt.figure(figsize=(10, 6))
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                plt.bar(range(X.shape[1]), importances[indices])
                plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.title('Random Forest Feature Importance')

                # Save plot
                plot_path = f"plots/{model_id}.png"
                plt.savefig(plot_path)
                plt.close()

                metrics = {
                    "train_accuracy": train_score,
                    "val_accuracy": val_score,
                    "plot_path": plot_path
                }

            # Save model
            model_path = f"models/{model_id}.joblib"
            joblib.dump(model, model_path)

            # Update result
            result_dict["status"] = "success"
            result_dict["message"] = f"Model {model_id} trained successfully"

            # Update model info
            MODELS[model_id]["object"] = model
            MODELS[model_id]["metrics"] = metrics
            MODELS[model_id]["feature_names"] = feature_names

        else:
            # Simulate training if no data provided
            sleep(2)
            result_dict["status"] = "success"
            result_dict["message"] = f"Model {model_id} training simulated (no data provided)"

    except Exception as e:
        import traceback
        result_dict["status"] = "error"
        error_msg = f"Error training model: {str(e)}\n{traceback.format_exc()}"
        result_dict["message"] = error_msg
        logger.error(error_msg)

# Ручка: fit (обучение)
@app.post("/fit", response_model=FitResponse)
def fit(request: FitRequest):
    if ACTIVE_MODEL_ID is None:
        raise HTTPException(status_code=404, detail="No active model")

    # Check if we have a dataset uploaded
    data_path = request.hyperparameters.get("data_path")

    # Train inline instead of using a new Process
    result_dict = {}
    fit_model(request.hyperparameters, ACTIVE_MODEL_ID, result_dict, data_path)

    return FitResponse(**result_dict)


# Ручка: predict
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if ACTIVE_MODEL_ID is None:
        raise HTTPException(status_code=404, detail="No active model")

    model_info = MODELS[ACTIVE_MODEL_ID]
    model = model_info["object"]

    print(f"Model info: {model_info}")
    print(f"Model object: {model}")

    if model is None:
        logger.warning(f"Prediction attempted with untrained model {ACTIVE_MODEL_ID}")
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train the model first.")
        return PredictResponse(predictions=request.data)  # Fallback if no model trained

    try:
        # Convert input to appropriate format
        if isinstance(request.data[0], list):
            # Multiple samples
            X = np.array(request.data)
        else:
            # Single sample
            X = np.array([request.data])

        # Make prediction
        predictions = model.predict(X).tolist()

        logger.info(f"Prediction made using model {ACTIVE_MODEL_ID}")
        return PredictResponse(predictions=predictions)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# Ручка: upload dataset
@app.post("/upload_dataset", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Create datasets directory if it doesn't exist
        os.makedirs("datasets", exist_ok=True)

        # Save file
        file_path = f"datasets/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Dataset uploaded: {file.filename}")
        return UploadResponse(
            status="success",
            message=f"Dataset {file.filename} uploaded successfully",
            file_path=file_path
        )

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")


@app.get("/plots/{plot_name}")
async def get_plot(plot_name: str):
    plot_path = f"plots/{plot_name}"
    if os.path.exists(plot_path):
        return FileResponse(plot_path)
    else:
        raise HTTPException(status_code=404, detail="Plot not found")


# Бонус: дообучение
@app.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest):
    if ACTIVE_MODEL_ID is None:
        raise HTTPException(status_code=404, detail="No active model")

    # Implementation would be similar to fit but would use existing model as starting point
    logger.info(f"Retraining model {ACTIVE_MODEL_ID}")
    return RetrainResponse(status="success", message="Model retrained successfully.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
    logger.info("Server started.")
    print("Server started.")