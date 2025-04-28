from typing import List, Optional
from pydantic import BaseModel, Field, Annotated

class ModelInfo(BaseModel):
    id: Annotated[str, Field(description="Model ID")]
    name: Annotated[str, Field(description="Model name")]
    description: Annotated[str, Field(description="Model description")]
    is_active: Annotated[bool, Field(description="Is model active")]

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

class RetrainRequest(BaseModel):
    hyperparameters: Annotated[dict, Field(description="Hyperparameters for retraining")]

class RetrainResponse(BaseModel):
    status: Annotated[str, Field(description="Retraining status")]
    message: Annotated[str, Field(description="Details")]
