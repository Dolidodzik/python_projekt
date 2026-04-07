import pandas as pd


def get_columns(df: pd.DataFrame, columns: str):
    if columns:
        cols = [c.strip() for c in columns.split(",")]
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Column {c} not found")
        return cols
    return df.select_dtypes(include="number").columns.tolist()


def correlation(df: pd.DataFrame, method: str, columns: str = None):
    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError("Invalid method")

    cols = get_columns(df, columns)

    corr = df[cols].corr(method=method)

    return corr