import pandas as pd
from sklearn.preprocessing import LabelEncoder
from fastapi import HTTPException


def encode_data(df, method, columns=None):
    df = df.copy()

    method = method.lower()

    if method not in ["onehot", "label"]:
        raise HTTPException(status_code=400, detail="Invalid method")

    if columns:
        cols = columns
    else:
        cols = df.select_dtypes(include="object").columns.tolist()

    if not cols:
        raise HTTPException(
            status_code=400, detail="No categorical columns"
        )

    if method == "onehot":
        df = pd.get_dummies(df, columns=cols)

    else:
        for col in cols:
            le = LabelEncoder()
            df.loc[:, col] = le.fit_transform(
                df[col].astype(str).fillna("NaN")
            )

    return df