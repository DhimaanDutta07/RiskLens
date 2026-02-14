from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def get_model(name: str, scale_pos_weight: float = 1.0):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")

    if name == "xgb":
        if XGBClassifier is None:
            raise ImportError("pip install xgboost")

        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )

    raise ValueError("Invalid model name")
