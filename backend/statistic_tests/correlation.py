import pandas as pd
from typing import List

def get_columns(df: pd.DataFrame, columns: List[str]):
    if not columns:
        raise ValueError("Columns parameter is required")

    cols = [c.strip() for c in columns if c.strip()]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column {c} not found")

    return cols


def correlation(df: pd.DataFrame, method: str, columns: List[str]):
    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError("Invalid method")

    cols = get_columns(df, columns)

    return df[cols].corr(method=method)