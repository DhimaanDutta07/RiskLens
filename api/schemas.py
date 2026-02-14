from pydantic import BaseModel
from typing import Any, Dict, List


class PredictRequest(BaseModel):
    data: Dict[str, Any]


class PredictBatchRequest(BaseModel):
    data: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    proba: float
    pred: int


class PredictBatchResponse(BaseModel):
    probas: List[float]
    preds: List[int]