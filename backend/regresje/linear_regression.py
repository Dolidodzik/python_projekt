import statsmodels.api as sm
import pandas as pd
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional, List


def format_float_to_string(value):
    """Format a float to 6 decimal places as a string (no scientific notation)."""
    if value is None:
        return None
    return f"{value:.6f}"


def train_linear_regression(
    df: pd.DataFrame,
    mode: str,
    columns: Optional[str],
    target_column: str,
):
    # żeby trenować musimy miec przynajmniej 1 predyktor i 1 target
    if df.empty or len(df.columns) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="INVALID_DATASET",
        )

    # predyktory można wybrać na 3 sposoby: all_parameters, include_parameters, exclude_parameters
    allowed_modes = {"all_parameters", "include_parameters", "exclude_parameters"}
    if mode not in allowed_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="INVALID_MODE",
        )

    # wybrana kolumna musi byc w dfie
    if target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="TARGET_COLUMN_NOT_FOUND",
        )

    # domyślnie wybieramy sobie wszystkie kolumny oprócz targeta jako predyktory
    all_feature_cols: List[str] = [c for c in df.columns if c != target_column]

    if mode == "all_parameters":
        feature_cols = all_feature_cols
    else:
        # jeżeli all_parameters nie jest true, to trzeba podac jakies kolumny (i dla include_parameters i dla exclude_parameters)
        if not columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="COLUMNS_REQUIRED",
            )

        # robimy ze stringa listę stringow podzielona przecinkiem
        requested = [c.strip() for c in columns.split(",") if c.strip()]

        # sprawdzamy czy wszystkie napewno sa w dfie
        unknown = [c for c in requested if c not in df.columns]
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="UNKNOWN_COLUMNS",
            )

        if mode == "include_parameters":
            # include_parameters - wybrane kolumny zostaja
            feature_cols = [c for c in requested if c != target_column]
        else:
            # exclude_parameters - wybrane kolumny sa usuwane
            feature_cols = [c for c in all_feature_cols if c not in requested]

    if not feature_cols:
        # jeżeli nie ma koncowo żadnych kolumn to da sie trenowac
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="NO_FEATURE_COLUMNS_SELECTED",
        )

    # upewniamy sie ze wszystkie kolumny sa numeryczne (cechy jakościowe powinny wcześniej zostać z-onehot-encodowane)
    numeric_cols = feature_cols + [target_column]
    non_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="NON_NUMERIC_COLUMNS_NOT_ALLOWED",
        )

    # upewniamy sie ze wszystkie kolumny sa bez nanow (wtedy trenowanie sie wywala)
    if df[numeric_cols].isna().any().any():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="NANS_NOT_ALLOWED",
        )

    # trenowanie prostej regresji liniowej
    X = df[feature_cols]
    y = df[target_column]
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # 95% przedziały pewności
    conf_int = model.conf_int(alpha=0.05)

    parameters = []

    # przeciecie z osia y to tez parameter
    parameters.append(
        {
            "name": "intercept",
            "coefficient": format_float_to_string(model.params.iloc[0]),
            "p_value": format_float_to_string(model.pvalues.iloc[0]),
            "standard_error": format_float_to_string(model.bse.iloc[0]),
            "conf_int_lower": format_float_to_string(conf_int.iloc[0, 0]),
            "conf_int_upper": format_float_to_string(conf_int.iloc[0, 1]),
        }
    )

    # parametry dla kazdego predyktora
    for i, feat in enumerate(feature_cols):
        parameters.append(
            {
                "name": feat,
                "coefficient": format_float_to_string(model.params.iloc[i + 1]),
                "p_value": format_float_to_string(model.pvalues.iloc[i + 1]),
                "standard_error": format_float_to_string(model.bse.iloc[i + 1]),
                "conf_int_lower": format_float_to_string(conf_int.iloc[i + 1, 0]),
                "conf_int_upper": format_float_to_string(conf_int.iloc[i + 1, 1]),
            }
        )

    # zwrotka
    response = {
        "parameters": parameters,
        "r_squared": format_float_to_string(model.rsquared),
        "adj_r_squared": format_float_to_string(model.rsquared_adj),
        "f_p_value": format_float_to_string(model.f_pvalue),
    }
    return JSONResponse(status_code=200, content=response)
