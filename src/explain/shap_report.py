import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def shap_report(
    model_path: str,
    data_path: str,
    out_dir: str = "artifacts/reports/shap",
    target_col: str = "is_fraud",
    sample_size: int = 2000
):
    os.makedirs(out_dir, exist_ok=True)

    pipe = joblib.load(model_path)
    df = pd.read_csv(data_path).dropna(subset=[target_col])
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)
        y = y.loc[X.index]

    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    X_t = pre.transform(X)
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(X_t.shape[1])])

    if hasattr(model, "get_booster") or model.__class__.__name__.lower().startswith("xgb"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_t)
    else:
        explainer = shap.LinearExplainer(model, X_t)
        sv = explainer.shap_values(X_t)

    sv_arr = sv[1] if isinstance(sv, list) else sv
    mean_abs = np.abs(sv_arr).mean(axis=0)

    top = (
        pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(30)
    )
    top.to_csv(os.path.join(out_dir, "top_features.csv"), index=False)

    plt.figure()
    shap.summary_plot(sv_arr, X_t, feature_names=feat_names, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary.png"), dpi=180, bbox_inches="tight")
    plt.close()

    return {"out_dir": out_dir, "top_features_path": os.path.join(out_dir, "top_features.csv")}
