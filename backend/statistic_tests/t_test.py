import pandas as pd
from scipy.stats import ttest_ind

def t_test(df: pd.DataFrame, group_col: str, value_col: str):
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("Column not found")

    groups = df[group_col].dropna().unique()

    if len(groups) != 2:
        raise ValueError("t-test requires exactly 2 groups")

    group1 = df[df[group_col] == groups[0]][value_col].dropna()
    group2 = df[df[group_col] == groups[1]][value_col].dropna()

    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("Each group must have at least 2 observations")

    t_stat, p_value = ttest_ind(group1, group2)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value)
    }
