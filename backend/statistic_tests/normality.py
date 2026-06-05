import pandas as pd
from scipy.stats import shapiro

def normality_test(df: pd.DataFrame, value_col: str):
    if value_col not in df.columns:
        raise ValueError("Column not found")

    data = df[value_col].dropna()

    if len(data) < 3:
        raise ValueError("Not enough data for Shapiro test")

    stat, p_value = shapiro(data)

    return {
        "statistic": float(stat),
        "p_value": float(p_value)
    }
