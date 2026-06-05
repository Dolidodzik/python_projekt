import pandas as pd

def covariance(df: pd.DataFrame, col1: str, col2: str):
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Column not found")

    cov_matrix = df[[col1, col2]].cov()

    return {
        "covariance": float(cov_matrix.loc[col1, col2]),
    }
