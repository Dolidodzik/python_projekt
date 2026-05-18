from sklearn.preprocessing import MinMaxScaler, StandardScaler
from fastapi import HTTPException
from .utils import get_numeric_columns


def normalize_data(df, method, columns=None):
    df = df.copy()

    method = method.lower()

    if method not in ["minmax", "zscore"]:
        raise HTTPException(status_code=400, detail="Invalid method")

    numeric_cols = get_numeric_columns(df, columns)

    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaled = scaler.fit_transform(df[numeric_cols])
    df.loc[:, numeric_cols] = scaled

    return df