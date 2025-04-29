from typing import Annotated, Dict, Any
from pydantic import BaseModel, Field

class ModelInfo(BaseModel):
    id: Annotated[str, Field(description="ID модели")]
    name: Annotated[str, Field(description="Название модели")]
    description: Annotated[str, Field(description="Описание модели")]
    is_active: Annotated[bool, Field(description="Активна ли модель")]
    type: Annotated[str, Field(description="Тип модели (xgboost или randomforest)")] = "xgboost"
    metrics: Annotated[Dict[str, Any], Field(description="Метрики модели")] = {}

class FitRequest(BaseModel):
    hyperparameters: Annotated[dict, Field(description="Гиперпараметры для обучения")]

class FitResponse(BaseModel):
    status: Annotated[str, Field(description="Статус обучения")]
    message: Annotated[str, Field(description="Детали")]

class PredictRequest(BaseModel):
    data: Annotated[list, Field(description="Входные данные для предсказания")]

class PredictResponse(BaseModel):
    predictions: Annotated[list, Field(description="Предсказания модели")]

class SetModelRequest(BaseModel):
    id: Annotated[str, Field(description="ID модели, которую нужно сделать активной")]

class ModelCreationRequest(BaseModel):
    name: Annotated[str, Field(description="Название модели")]
    model_type: Annotated[str, Field(description="Тип модели (xgboost или randomforest)")]

class ModelCreationResponse(BaseModel):
    status: Annotated[str, Field(description="Статус создания")]
    message: Annotated[str, Field(description="Детали")]
    model_id: Annotated[str, Field(description="ID созданной модели")]

class UploadResponse(BaseModel):
    status: Annotated[str, Field(description="Статус загрузки")]
    message: Annotated[str, Field(description="Детали")]
    file_path: Annotated[str, Field(description="Путь к загруженному файлу")]

class RetrainRequest(BaseModel):
    hyperparameters: Annotated[dict, Field(description="Гиперпараметры для переобучения")]

class RetrainResponse(BaseModel):
    status: Annotated[str, Field(description="Статус переобучения")]
    message: Annotated[str, Field(description="Детали")]
