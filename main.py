import argparse
import pandas as pd

from src.data.clean import DataCleaner
from src.features.build import Features
from src.data.split import Splitter

from src.models.preprocessor import build_preprocessor, fit_transform_save, transform_save
from src.models.model import get_model
from src.models.trainer import Trainer

from src.explain.shap_report import shap_report
from src.monitoring.evidently_report import evidently_reports


DATA_RAW = "data/raw/transactions.csv"
DATA_FEATURES = "data/processed/fraud_features.csv"


def run_data():
    df = pd.read_csv(DATA_RAW)
    df = DataCleaner(out_path=DATA_FEATURES).clean(df)
    df = Features(out_path=DATA_FEATURES).build(df)
    return df


def run_train(model_name: str):
    df = pd.read_csv(DATA_FEATURES)
    df = df.dropna(subset=["is_fraud"])
    df["is_fraud"] = df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = Splitter().split(df)

    spw = 1.0
    if model_name == "xgb":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = float(neg / max(pos, 1))

    pre = build_preprocessor(X_train)

    fit_transform_save(pre, X_train, "data/processed/X_train_processed.csv")
    transform_save(pre, X_test, "data/processed/X_test_processed.csv")

    model = get_model(model_name, scale_pos_weight=spw)

    trainer = Trainer(
        pre=pre,
        model=model,
        model_path=f"artifacts/models/{model_name}.joblib",
        report_path=f"artifacts/metrics/{model_name}.json",
    )

    rep = trainer.fit(X_train, y_train, X_test, y_test, model_name=model_name)
    print(model_name, rep["roc_auc"], rep["pr_auc"])


def run_shap(model_name: str):
    shap_report(
        model_path=f"artifacts/models/{model_name}.joblib",
        data_path=DATA_FEATURES,
        out_dir="artifacts/shap",
        target_col="is_fraud",
        sample_size=2000,
    )


def run_monitor(model_name: str):
    evidently_reports(
        model_path=f"artifacts/models/{model_name}.joblib",
        reference_path=DATA_FEATURES,
        current_path=DATA_FEATURES,
        out_dir="artifacts/monitoring",
        target_col="is_fraud",
    )


def main(mode: str):
    if mode in ("all", "data"):
        run_data()

    if mode in ("all", "train"):
        run_train("logreg")
        run_train("xgb")

    if mode in ("all", "shap"):
        run_shap("xgb")

    if mode in ("all", "monitor"):
        run_monitor("xgb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", "data", "train", "shap", "monitor"])
    args = parser.parse_args()
    main(args.mode)
