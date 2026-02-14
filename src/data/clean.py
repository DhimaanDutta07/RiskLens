import os
import pandas as pd

class DataCleaner:
    def __init__(self, out_path="data/processed/fraud_clean.csv"):
        self.out_path = out_path

    def clean(self, df: pd.DataFrame):
        df = df.copy()

        df = df.drop(columns=["order_id", "user_id", "device_id", "city"], errors="ignore")

        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

        df = df.dropna(subset=["is_fraud"])

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        df.to_csv(self.out_path, index=False)

        return df
