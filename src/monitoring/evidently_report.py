import os
import joblib
import pandas as pd

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataSummaryPreset, DataDriftPreset, ClassificationPreset


def evidently_reports(model_path, reference_path, current_path, out_dir="artifacts/monitoring", target_col="is_fraud"):
    from evidently import BinaryClassification

    os.makedirs(out_dir, exist_ok=True)

    pipe = joblib.load(model_path)

    ref = pd.read_csv(reference_path).dropna(subset=[target_col]).copy()
    cur = pd.read_csv(current_path).dropna(subset=[target_col]).copy()

    def add_preds(df):
        X = df.drop(columns=[target_col])
        df[target_col] = df[target_col].astype(int)
        df["prediction_proba"] = pipe.predict_proba(X)[:, 1]
        df["prediction"] = (df["prediction_proba"] >= 0.5).astype(int)
        return df

    ref = add_preds(ref)
    cur = add_preds(cur)

    num_cols = [
        c for c in cur.columns
        if c not in [target_col, "prediction", "prediction_proba"]
        and pd.api.types.is_numeric_dtype(cur[c])
    ]
    cat_cols = [
        c for c in cur.columns
        if c not in [target_col, "prediction", "prediction_proba"]
        and not pd.api.types.is_numeric_dtype(cur[c])
    ]

    definition = DataDefinition(
        numerical_columns=num_cols + [target_col, "prediction", "prediction_proba"],
        categorical_columns=cat_cols,
        classification=[BinaryClassification(
            target=target_col,
            prediction_labels="prediction",
            prediction_probas="prediction_proba",
            pos_label=1
        )]
    )

    ref_ds = Dataset.from_pandas(ref, data_definition=definition)
    cur_ds = Dataset.from_pandas(cur, data_definition=definition)

    rep = Report([DataSummaryPreset(), DataDriftPreset(), ClassificationPreset()])
    snap = rep.run(cur_ds, ref_ds)

    out_path = os.path.join(out_dir, "evidently_report.html")
    snap.save_html(out_path)
    return out_path
