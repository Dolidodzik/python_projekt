import pandas as pd
from scipy.stats import chi2_contingency

def chi_square(df: pd.DataFrame, col1: str, col2: str):
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Column not found")

    contingency_table = pd.crosstab(df[col1], df[col2])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "expected": expected.tolist()
    }
