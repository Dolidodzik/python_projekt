import pandas as pd
from scipy.stats import mannwhitneyu

def mann_whitney_test(df: pd.DataFrame, group_col: str, value_col: str):
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("Column not found")

    groups = df[group_col].dropna().unique()

    if len(groups) != 2:
        raise ValueError("Requires exactly 2 groups")

    g1 = df[df[group_col] == groups[0]][value_col].dropna()
    g2 = df[df[group_col] == groups[1]][value_col].dropna()

    stat, p_value = mannwhitneyu(g1, g2)

    return {
        "statistic": float(stat),
        "p_value": float(p_value)
    }