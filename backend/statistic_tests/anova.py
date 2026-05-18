import pandas as pd
from scipy.stats import f_oneway


def anova(df: pd.DataFrame, group_col: str, value_col: str):
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("Column not found")

    groups = df[group_col].dropna().unique()

    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")

    samples = [
        df[df[group_col] == g][value_col].dropna()
        for g in groups
    ]

    f_stat, p_value = f_oneway(*samples)

    return {
        "f_value": float(f_stat),
        "p_value": float(p_value)
    }