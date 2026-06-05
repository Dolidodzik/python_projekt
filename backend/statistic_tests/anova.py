import pandas as pd
from scipy.stats import f_oneway
import numpy as np

def anova(df: pd.DataFrame, group_col: str, value_col: str):
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("Column not found")

    groups = df[group_col].dropna().unique()

    if len(groups) < 2:
        raise ValueError(f"anova requires at least 2 groups, found {len(groups)}")

    samples = [
        df[df[group_col] == g][value_col].dropna()
        for g in groups
    ]

    sample_sizes = [len(sample) for sample in samples]
    if min(sample_sizes) < 2:
        raise ValueError(f"Each group must have at least 2 samples, got sizes {sample_sizes}")

    f_stat, p_value = f_oneway(*samples)

    if np.isnan(f_stat) or np.isnan(p_value):
        raise ValueError("Cannot compute anova - insufficient data")

    return {
        "f_value": float(f_stat),
        "p_value": float(p_value)
    }