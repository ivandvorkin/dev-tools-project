import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import *
from typing import Dict
from time import sleep
import multiprocessing
import uuid

app = FastAPI(title="ML Model Management API")

# Логирование с ротацией
log_handler = RotatingFileHandler("app/logs/server.log", maxBytes=1000000, backupCount=5)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
logger = logging.getLogger("uvicorn")
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

# CORS для фронта
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели (заглушки)
MODELS: Dict[str, Dict] = {}
ACTIVE_MODEL_ID: Optional[str] = None

# Загрузка модели при запуске
@app.on_event("startup")
def load_models():
    global MODELS, ACTIVE_MODEL_ID
    # Пример: одна заглушка модели
    model_id = str(uuid.uuid4())
    MODELS[model_id] = {
        "id": model_id,
        "name": "DemoModel",
        "description": "A demo model",
        "is_active": True,
        "object": None  # Здесь будет объект модели
    }
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

# Вспомогательная функция для обучения
def fit_model(hyperparameters, model_id, result_dict):
    sleep_time = hyperparameters.get("sleep", 2)
    sleep(sleep_time)  # имитируем долгое обучение
    result_dict["status"] = "success"
    result_dict["message"] = f"Model {model_id} trained in {sleep_time} seconds"

# Ручка: fit (обучение)
@app.post("/fit", response_model=FitResponse)
def fit(request: FitRequest):
    if ACTIVE_MODEL_ID is None:
        raise HTTPException(status_code=404, detail="No active model")
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    p = multiprocessing.Process(target=fit_model, args=(request.hyperparameters, ACTIVE_MODEL_ID, result_dict))
    p.start()
    p.join(timeout=10)
    if p.is_alive():
        p.terminate()
        logger.warning("Training killed due to timeout.")
        return FitResponse(status="timeout", message="Training exceeded 10 seconds and was stopped.")
    return FitResponse(**result_dict)

# Ручка: predict
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if ACTIVE_MODEL_ID is None:
        raise HTTPException(status_code=404, detail="No active model")
    # Заглушка: просто возвращаем входные данные
    logger.info(f"Prediction requested using model {ACTIVE_MODEL_ID}")
    return PredictResponse(predictions=request.data)

# Бонус: дообучение
@app.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest):
    if ACTIVE_MODEL_ID is None:
        raise HTTPException(status_code=404, detail="No active model")
    # Имитация дообучения
    logger.info(f"Retraining model {ACTIVE_MODEL_ID}")
    return RetrainResponse(status="success", message="Model retrained successfully.")
