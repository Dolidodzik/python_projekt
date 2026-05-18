import pandas as pd

def covariance(df: pd.DataFrame, columns: list):
    if not columns or len(columns) < 2:
        raise ValueError("At least 2 columns required")

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found")

    cov_matrix = df[columns].cov()

    return cov_matrix