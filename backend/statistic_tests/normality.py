import pandas as pd
from scipy.stats import shapiro


def normality_test(df: pd.DataFrame, column: str):
    if column not in df.columns:
        raise ValueError("Column not found")

    data = df[column].dropna()

    if len(data) < 3:
        raise ValueError("Not enough data for Shapiro test")

    stat, p_value = shapiro(data)

    return {
        "statistic": float(stat),
        "p_value": float(p_value)
    }