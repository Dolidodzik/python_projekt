def format_float_to_string(value):
    if value is None:
        return None
    return f"{value:.6f}"


def adjusted_r_squared_from_r2(r2: float, n: int, p: int):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def iqr_outliers_count(df, cols):
    if df is None or len(cols) == 0 or len(df) == 0:
        return 0
    mask = None
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        m = (df[c] < (q1 - 1.5 * iqr)) | (df[c] > (q3 + 1.5 * iqr))
        mask = m if mask is None else (mask | m)
    return int(mask.sum()) if mask is not None else 0


def vif_dict(df, cols):
    try:
        import numpy as np
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X = df[cols]
        X = sm.add_constant(X, has_constant="add")
        values = X.values.astype(float)
        names = list(X.columns)
        out = {}
        for i, name in enumerate(names):
            if name == "const":
                continue
            v = float(variance_inflation_factor(values, i))
            out[name] = float(np.round(v, 6)) if np.isfinite(v) else v
        return out
    except Exception:
        return {}
