import pandas as pd
from sklearn.impute import KNNImputer
from fastapi import HTTPException
from .utils import get_numeric_columns


def impute_data(df, method, columns=None):
    df = df.copy()

    method = method.lower()

    if method not in ["median", "knn"]:
        raise HTTPException(status_code=400, detail="Invalid method")

    numeric_cols = get_numeric_columns(df, columns)

    if method == "median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

    elif method == "knn":
        imputer = KNNImputer()
        imputed = imputer.fit_transform(df[numeric_cols])
        df.loc[:, numeric_cols] = imputed

    return df