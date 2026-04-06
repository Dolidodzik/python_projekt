import numpy as np
from fastapi import HTTPException
from .utils import get_numeric_columns


def remove_outliers_data(df, method, columns=None, threshold=None):
    df = df.copy()

    method = method.lower()

    if method not in ["iqr", "zscore"]:
        raise HTTPException(status_code=400, detail="Invalid method")

    numeric_cols = get_numeric_columns(df, columns)

    mask = np.ones(len(df), dtype=bool)

    if method == "iqr":
        if threshold is None:
            threshold = 1.5

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR

            col_mask = (df[col] >= lower) & (df[col] <= upper)
            mask &= col_mask

    else:
        if threshold is None:
            threshold = 3

        for col in numeric_cols:
            std = df[col].std()

            if std == 0 or np.isnan(std):
                continue

            z = (df[col] - df[col].mean()) / std
            col_mask = np.abs(z) < threshold
            mask &= col_mask

    return df[mask].reset_index(drop=True)