from typing import List, Optional, Annotated, Dict, Any
from pydantic import BaseModel, Field

class ModelInfo(BaseModel):
    id: Annotated[str, Field(description="Model ID")]
    name: Annotated[str, Field(description="Model name")]
    description: Annotated[str, Field(description="Model description")]
    is_active: Annotated[bool, Field(description="Is model active")]
    type: Annotated[str, Field(description="Model type (xgboost or randomforest)")] = "xgboost"
    metrics: Annotated[Dict[str, Any], Field(description="Model metrics")] = {}

class FitRequest(BaseModel):
    hyperparameters: Annotated[dict, Field(description="Hyperparameters for training")]

class FitResponse(BaseModel):
    status: Annotated[str, Field(description="Training status")]
    message: Annotated[str, Field(description="Details")]

class PredictRequest(BaseModel):
    data: Annotated[list, Field(description="Input data for prediction")]

class PredictResponse(BaseModel):
    predictions: Annotated[list, Field(description="Model predictions")]

class SetModelRequest(BaseModel):
    id: Annotated[str, Field(description="ID of the model to set active")]

class ModelCreationRequest(BaseModel):
    name: Annotated[str, Field(description="Model name")]
    model_type: Annotated[str, Field(description="Model type (xgboost or randomforest)")]

class ModelCreationResponse(BaseModel):
    status: Annotated[str, Field(description="Creation status")]
    message: Annotated[str, Field(description="Details")]
    model_id: Annotated[str, Field(description="ID of the created model")]

class UploadResponse(BaseModel):
    status: Annotated[str, Field(description="Upload status")]
    message: Annotated[str, Field(description="Details")]
    file_path: Annotated[str, Field(description="Path to the uploaded file")]

class RetrainRequest(BaseModel):
    hyperparameters: Annotated[dict, Field(description="Hyperparameters for retraining")]

class RetrainResponse(BaseModel):
    status: Annotated[str, Field(description="Retraining status")]
    message: Annotated[str, Field(description="Details")]