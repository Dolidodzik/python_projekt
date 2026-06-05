import pandas as pd

def correlation(df: pd.DataFrame, col1: str, col2: str, method: str):
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Column not found")

    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError("Invalid method")

    corr_matrix = df[[col1, col2]].corr(method=method)

    return {
        "correlation": float(corr_matrix.loc[col1, col2]),
    }
