import os
import joblib
import mlflow

from sklearn.pipeline import Pipeline
from .evaluate import evaluate, save_report, save_graphs


class Trainer:
    def __init__(self, pre, model, model_path: str, report_path: str):
        self.model_path = model_path
        self.report_path = report_path
        self.pipe = Pipeline([("pre", pre), ("model", model)])

    def fit(self, X_train, y_train, X_test, y_test, model_name: str = "model") -> dict:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "RiskLens"))

        with mlflow.start_run(run_name=model_name):
            self.pipe.fit(X_train, y_train)

            y_prob = self.pipe.predict_proba(X_test)[:, 1]
            rep = evaluate(y_test, y_prob)

            joblib.dump(self.pipe, self.model_path)
            save_report(rep, self.report_path)

            save_graphs(
                y_test, y_prob, rep,
                out_dir="artifacts/graphs",
                prefix=os.path.splitext(os.path.basename(self.model_path))[0]
            )

            mlflow.log_metrics({
                "roc_auc": rep["roc_auc"],
                "pr_auc": rep["pr_auc"],
                "threshold": rep["threshold"],
            })

            mlflow.log_artifact(self.model_path, artifact_path="models")
            mlflow.log_artifact(self.report_path, artifact_path="metrics")

            for fname in os.listdir("artifacts/graphs"):
                if fname.startswith(os.path.splitext(os.path.basename(self.model_path))[0]):
                    mlflow.log_artifact(os.path.join("artifacts/graphs", fname), artifact_path="graphs")

        return rep
