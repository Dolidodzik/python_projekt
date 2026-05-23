import pandas as pd
from scipy.stats import mannwhitneyu

def mann_whitney_test(df: pd.DataFrame, group_col: str, value_col: str):
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("Column not found")

    groups = df[group_col].dropna().unique()

    if len(groups) != 2:
        raise ValueError("Mann-Whitney test requires exactly 2 groups")

    group1 = df[df[group_col] == groups[0]][value_col].dropna()
    group2 = df[df[group_col] == groups[1]][value_col].dropna()

    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("Each group must have at least 2 observations")

    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_value)
    }