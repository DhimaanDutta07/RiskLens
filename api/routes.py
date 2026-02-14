import os
import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .schemas import PredictRequest, PredictBatchRequest, PredictResponse, PredictBatchResponse

router = APIRouter()

MODEL = None
THRESHOLD = 0.5
EXPECTED_COLS = None

RATES_TO_INR = {
    "INR": 1.0,
    "USD": 83.0,
    "EUR": 90.0,
    "GBP": 105.0,
    "BRL": 16.5,
    "MXN": 4.9,
    "NGN": 0.054,
    "PHP": 1.48,
    "IDR": 0.0053,
}

RAW_BASE_COLS = [
    "event_time",
    "order_amount",
    "currency",
    "country",
    "payment_method",
    "item_count",
    "ip_risk",
    "device_risk",
    "billing_shipping_mismatch",
    "shipping_address_changed",
    "email_verified",
    "account_age_days",
    "orders_last_7d",
    "failed_payments_last_24h",
    "distance_ip_to_shipping_km",
]

ENGINEERED_COLS = [
    "event_hour",
    "unusual_time",
    "amount_inr",
    "unusual_amount",
    "unusual_item_qty",
    "ip_risk_score",
    "shipping_risk_score",
    "new_account",
    "unusual_distance",
    "suspected_method",
    "unusual_orders_last_7d",
    "suspected_failed_payments_last_24h",
    "overall_risk_score",
]

ALL_FEATURE_COLS = RAW_BASE_COLS + ENGINEERED_COLS


def load_model(model_path: str):
    global MODEL, EXPECTED_COLS
    MODEL = joblib.load(model_path)
    try:
        EXPECTED_COLS = list(MODEL.named_steps["pre"].feature_names_in_)
    except Exception:
        EXPECTED_COLS = None


def _align_columns(X: pd.DataFrame) -> pd.DataFrame:
    if EXPECTED_COLS is None:
        return X
    for c in EXPECTED_COLS:
        if c not in X.columns:
            X[c] = np.nan
    return X[EXPECTED_COLS]


def _as_int(x, default=0) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, bool):
            return int(x)
        return int(float(x))
    except Exception:
        return int(default)


def _as_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _parse_event_time(v) -> pd.Timestamp:
    if v is None or (isinstance(v, str) and not v.strip()):
        raise ValueError("event_time is required")
    return pd.to_datetime(v, errors="raise")


def _build_features(d: dict) -> dict:
    d = dict(d)

    d["event_time"] = _parse_event_time(d.get("event_time"))
    d["currency"] = str(d.get("currency") or "INR").upper().strip()
    d["country"] = str(d.get("country") or "IN").upper().strip()
    d["payment_method"] = str(d.get("payment_method") or "card").lower().strip()

    d["order_amount"] = _as_float(d.get("order_amount"), 0.0)
    d["item_count"] = _as_int(d.get("item_count"), 1)

    d["ip_risk"] = _as_int(d.get("ip_risk"), 0)
    d["device_risk"] = _as_int(d.get("device_risk"), 0)

    d["billing_shipping_mismatch"] = _as_int(d.get("billing_shipping_mismatch"), 0)
    d["shipping_address_changed"] = _as_int(d.get("shipping_address_changed"), 0)

    d["email_verified"] = _as_int(d.get("email_verified"), 1)
    d["account_age_days"] = _as_int(d.get("account_age_days"), 9999)

    d["orders_last_7d"] = _as_int(d.get("orders_last_7d"), 0)
    d["failed_payments_last_24h"] = _as_int(d.get("failed_payments_last_24h"), 0)

    d["distance_ip_to_shipping_km"] = _as_float(d.get("distance_ip_to_shipping_km"), 0.0)

    d["event_hour"] = int(pd.to_datetime(d["event_time"]).hour)
    d["unusual_time"] = int(d["event_hour"] >= 23 or d["event_hour"] <= 4)

    rate = RATES_TO_INR.get(d["currency"], 1.0)
    d["amount_inr"] = float(d["order_amount"] * rate)

    d["unusual_amount"] = int(d["amount_inr"] >= 11000)
    d["unusual_item_qty"] = int(d["item_count"] >= 3)

    d["ip_risk_score"] = int(max(d["ip_risk"], d["device_risk"]))
    d["shipping_risk_score"] = int(max(d["billing_shipping_mismatch"], d["shipping_address_changed"]))

    d["new_account"] = int(d["account_age_days"] <= 740)
    d["unusual_distance"] = int(d["distance_ip_to_shipping_km"] >= 500)

    pm = str(d["payment_method"]).lower()
    d["suspected_method"] = int(("card" in pm) or ("wallet" in pm))

    d["unusual_orders_last_7d"] = int(d["orders_last_7d"] >= 20)
    d["suspected_failed_payments_last_24h"] = int(d["failed_payments_last_24h"] >= 2)

    d["overall_risk_score"] = int(
        d["unusual_time"]
        + d["unusual_amount"]
        + d["unusual_item_qty"]
        + d["ip_risk_score"]
        + d["shipping_risk_score"]
        + d["new_account"]
        + d["unusual_distance"]
        + d["suspected_method"]
        + d["unusual_orders_last_7d"]
        + d["suspected_failed_payments_last_24h"]
    )

    return d


def _to_df(features: dict) -> pd.DataFrame:
    row = {c: features.get(c, np.nan) for c in ALL_FEATURE_COLS}
    X = pd.DataFrame([row])
    X["event_time"] = pd.to_datetime(X["event_time"], errors="coerce")
    return X


@router.get("/")
def home():
    path = os.path.join("frontend", "index.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(path)


@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@router.get("/expected_columns")
def expected_columns():
    return {"expected_columns": EXPECTED_COLS or []}


@router.post("/debug_features")
def debug_features(payload: PredictRequest):
    feats = _build_features(payload.data)
    return {"features": {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in feats.items()}}


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        feats = _build_features(payload.data)
        X = _to_df(feats)
        X = _align_columns(X)

        proba = float(MODEL.predict_proba(X)[:, 1][0])
        pred = int(proba >= THRESHOLD)
        return {"proba": proba, "pred": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(payload: PredictBatchRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        feats_list = [_build_features(d) for d in payload.data]
        X = pd.DataFrame([{c: f.get(c, np.nan) for c in ALL_FEATURE_COLS} for f in feats_list])
        X["event_time"] = pd.to_datetime(X["event_time"], errors="coerce")
        X = _align_columns(X)

        probas = MODEL.predict_proba(X)[:, 1].astype(float).tolist()
        preds = [int(p >= THRESHOLD) for p in probas]
        return {"probas": probas, "preds": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))