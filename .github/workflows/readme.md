# RiskLens — Fraud Guard 🛡️

RiskLens is an end-to-end, production-style **fraud detection system** built to simulate a real fintech risk engine. It covers the full ML workflow: data → feature engineering → training → evaluation → explainability → monitoring → API → interactive frontend.

This project is designed to be **action-based**, meaning the model is trained on transaction behavior signals and outputs a fraud probability that can be mapped to operational decisions (allow / review / block).

---

## ✨ Highlights

- INR-only fraud scoring pipeline (consistent feature distribution)
- Feature engineering system (fraud signals + composite risk score)
- Baseline model: Logistic Regression
- Main model: XGBoost
- Evaluation artifacts:
  - ROC curve, PR curve, confusion matrix
  - threshold selection (best-F1)
  - JSON report saved per model
- Explainability:
  - SHAP summary plot
  - top features exported as CSV
- Monitoring:
  - Evidently report for drift + classification performance
- Deployment-ready:
  - FastAPI inference API with CORS
  - Modern dark UI fraud console (interactive scenario builder)
- MLOps:
  - DVC pipeline (`dvc.yaml`)
  - MLflow experiment tracking
  - Docker support
  - GitHub Actions CI

---

## 📁 Project Structure

RiskLens--Fraud-Guard/
├─ api/ # FastAPI inference service
├─ src/
│ ├─ data/ # cleaning + splitting
│ ├─ features/ # feature engineering
│ ├─ models/ # preprocessing + training + evaluation
│ ├─ explain/ # SHAP report
│ └─ monitoring/ # Evidently report
├─ data/
│ ├─ raw/ # raw dataset (DVC recommended)
│ └─ processed/ # engineered dataset (ignored in git)
├─ artifacts/
│ ├─ models/ # trained pipelines (.joblib)
│ ├─ metrics/ # evaluation JSON reports
│ ├─ graphs/ # ROC/PR/confusion/metric plots
│ ├─ shap/ # SHAP summary + top features
│ └─ monitoring/ # Evidently HTML report
├─ frontend/ # interactive fraud console
├─ main.py # orchestrator (data/train/shap/monitor)
├─ dvc.yaml # reproducible pipeline stages
├─ Dockerfile
└─ requirements.txt


---

## 🔢 Features Used

### Raw input features
- `event_time`
- `order_amount`
- `item_count`
- `ip_risk`, `device_risk`
- `billing_shipping_mismatch`, `shipping_address_changed`
- `email_verified`
- `account_age_days`
- `orders_last_7d`
- `failed_payments_last_24h`
- `distance_ip_to_shipping_km`
- `country`
- `payment_method`

### Engineered fraud signals
- `event_hour`, `unusual_time`
- `amount_inr`, `unusual_amount`
- `unusual_item_qty`
- `ip_risk_score`
- `shipping_risk_score`
- `new_account`
- `unusual_distance`
- `suspected_method`
- `unusual_orders_last_7d`
- `suspected_failed_payments_last_24h`
- `overall_risk_score`

---

## ⚙️ Setup

### Create virtual environment
```bash
python -m venv .venv
Activate
Windows

. .venv/Scripts/activate
Linux / Mac

source .venv/bin/activate
Install dependencies
pip install -r requirements.txt
🚀 Run Pipeline
Run everything
python main.py --mode all
Run step-by-step
python main.py --mode data
python main.py --mode train
python main.py --mode shap
python main.py --mode monitor
📊 Outputs (Artifacts)
After training, you will get:

Models
artifacts/models/logreg.joblib

artifacts/models/xgb.joblib

Metrics
artifacts/metrics/logreg.json

artifacts/metrics/xgb.json

Graphs
artifacts/graphs/*roc_curve.png

artifacts/graphs/*pr_curve.png

artifacts/graphs/*confusion.png

artifacts/graphs/*metrics_bar.png

SHAP
artifacts/shap/summary.png

artifacts/shap/top_features.csv

Monitoring
artifacts/monitoring/evidently_report.html

🧠 MLflow Tracking
Training logs runs into ./mlruns automatically.

Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5001
Open:

http://127.0.0.1:5001
🌐 Run the API (FastAPI)
Start server
uvicorn api.main:app --reload --port 8000
API base:

http://127.0.0.1:8000
🧪 Test with Postman
Health
GET

http://127.0.0.1:8000/health
Predict
POST

http://127.0.0.1:8000/predict
Body:

{
  "data": {
    "event_time": "2026-02-13 01:12:00",
    "order_amount": 47133,
    "currency": "INR",
    "country": "NG",
    "payment_method": "wallet",
    "item_count": 5,
    "ip_risk": 1,
    "device_risk": 1,
    "billing_shipping_mismatch": 1,
    "shipping_address_changed": 1,
    "email_verified": 0,
    "account_age_days": 5,
    "orders_last_7d": 44,
    "failed_payments_last_24h": 6,
    "distance_ip_to_shipping_km": 10210
  }
}
Debug engineered features
POST

http://127.0.0.1:8000/debug_features
🖥️ Frontend Console
Open:

frontend/index.html
The console:

lets you create high-risk scenarios

computes engineered columns automatically

sends full payload to the API

displays probability + risk band

🧩 Reproducibility with DVC
Run pipeline using DVC:

dvc repro
🐳 Docker
Build:

docker build -t risklens .
Run:

docker run -p 8000:8000 risklens
🔒 Notes
data/processed/, artifacts/, and mlruns/ should be ignored in git.

For large datasets, DVC is recommended.

Fraud probabilities may not always reach 0.9+ due to class imbalance; decision-making should use tuned thresholds.

📌 Author
Dhimaan Dutta
ML Engineering • Backend • MLOps • Fraud/Risk Systems

