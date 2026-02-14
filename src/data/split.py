import pandas as pd
from sklearn.model_selection import train_test_split

class Splitter:
    def split(self, df: pd.DataFrame):
        df = df.copy()

        X = df.drop(columns=["is_fraud"])
        y = df["is_fraud"]

        return train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
