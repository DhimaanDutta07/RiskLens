import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def fit_transform_save(pre: ColumnTransformer, X: pd.DataFrame, out_path: str) -> pd.DataFrame:
    Xt = pre.fit_transform(X)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cols = []
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    cols = num_cols + cat_cols

    df_out = pd.DataFrame(Xt, columns=cols)
    df_out.to_csv(out_path, index=False)

    return df_out


def transform_save(pre: ColumnTransformer, X: pd.DataFrame, out_path: str) -> pd.DataFrame:
    Xt = pre.transform(X)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    cols = num_cols + cat_cols

    df_out = pd.DataFrame(Xt, columns=cols)
    df_out.to_csv(out_path, index=False)

    return df_out
