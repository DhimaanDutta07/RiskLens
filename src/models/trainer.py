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

    def fit(self, X_train, y_train, X_test, y_test) -> dict:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("risklens-fraud-guard")

        model_name = os.path.splitext(os.path.basename(self.model_path))[0]

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)

            self.pipe.fit(X_train, y_train)

            y_prob = self.pipe.predict_proba(X_test)[:, 1]
            rep = evaluate(y_test, y_prob)

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

            joblib.dump(self.pipe, self.model_path)
            save_report(rep, self.report_path)

            save_graphs(
                y_test,
                y_prob,
                rep,
                out_dir="artifacts/graphs",
                prefix=model_name,
            )

            mlflow.log_metric("roc_auc", float(rep["roc_auc"]))
            mlflow.log_metric("pr_auc", float(rep["pr_auc"]))
            mlflow.log_metric("threshold", float(rep["threshold"]))

            mlflow.log_artifact(self.model_path)
            mlflow.log_artifact(self.report_path)

            graphs_dir = "artifacts/graphs"
            if os.path.isdir(graphs_dir):
                for fn in os.listdir(graphs_dir):
                    if fn.startswith(model_name + "_") and fn.endswith(".png"):
                        mlflow.log_artifact(os.path.join(graphs_dir, fn))

            return rep