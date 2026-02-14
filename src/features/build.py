import os
import pandas as pd

class Features:
    def __init__(self, out_path="data/processed/fraud_features.csv"):
        self.out_path = out_path

    def build(self, df: pd.DataFrame):
        df = df.copy()

        df["event_hour"] = df["event_time"].dt.hour.astype(int)
        df["unusual_time"] = ((df["event_hour"] >= 23) | (df["event_hour"] <= 4)).astype(int)

        rates_to_inr = {
            "INR": 1.0,
            "USD": 83.0,
            "EUR": 90.0,
            "GBP": 105.0,
            "BRL": 16.5,
            "MXN": 4.9,
            "NGN": 0.054,
            "PHP": 1.48,
            "IDR": 0.0053
        }

        df["amount_inr"] = df["order_amount"] * df["currency"].map(rates_to_inr)

        df["unusual_amount"] = (df["amount_inr"] >= 11000).astype(int)
        df["unusual_item_qty"] = (df["item_count"] >= 3).astype(int)

        df["ip_risk_score"] = df[["ip_risk", "device_risk"]].max(axis=1)

        df["shipping_risk_score"] = df[["billing_shipping_mismatch", "shipping_address_changed"]].max(axis=1)

        df["new_account"] = (df["account_age_days"] <= 740).astype(int)
        df["unusual_distance"] = (df["distance_ip_to_shipping_km"] >= 500).astype(int)

        pm = df["payment_method"].astype(str).str.lower()
        df["suspected_method"] = (pm.str.contains("card") | pm.str.contains("wallet")).astype(int)

        df["unusual_orders_last_7d"] = (df["orders_last_7d"] >= 20).astype(int)
        df["suspected_failed_payments_last_24h"] = (df["failed_payments_last_24h"] >= 2).astype(int)

        df["overall_risk_score"] = (
            df["unusual_time"]
            + df["unusual_amount"]
            + df["unusual_item_qty"]
            + df["ip_risk_score"]
            + df["shipping_risk_score"]
            + df["new_account"]
            + df["unusual_distance"]
            + df["suspected_method"]
            + df["unusual_orders_last_7d"]
            + df["suspected_failed_payments_last_24h"]
        )

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        df.to_csv(self.out_path, index=False)

        return df
