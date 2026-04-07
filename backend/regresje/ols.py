import statsmodels.api as sm
import pandas as pd
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from sklearn.metrics import mean_squared_error, r2_score
from typing import Optional, List

from . import utils


def run_ols(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feature_cols: List[str],
    target_column: str,
):
    # trenowanie prostej regresji liniowej
    X = train_df[feature_cols]
    y = train_df[target_column]
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # 95% przedziały pewności
    conf_int = model.conf_int(alpha=0.05)

    parameters = []

    # przeciecie z osia y to tez parameter
    parameters.append(
        {
            "name": "intercept",
            "coefficient": utils.format_float_to_string(model.params.iloc[0]),
            "p_value": utils.format_float_to_string(model.pvalues.iloc[0]),
            "standard_error": utils.format_float_to_string(model.bse.iloc[0]),
            "conf_int_lower": utils.format_float_to_string(conf_int.iloc[0, 0]),
            "conf_int_upper": utils.format_float_to_string(conf_int.iloc[0, 1]),
        }
    )

    # parametry dla kazdego predyktora
    for i, feat in enumerate(feature_cols):
        parameters.append(
            {
                "name": feat,
                "coefficient": utils.format_float_to_string(model.params.iloc[i + 1]),
                "p_value": utils.format_float_to_string(model.pvalues.iloc[i + 1]),
                "standard_error": utils.format_float_to_string(model.bse.iloc[i + 1]),
                "conf_int_lower": utils.format_float_to_string(conf_int.iloc[i + 1, 0]),
                "conf_int_upper": utils.format_float_to_string(conf_int.iloc[i + 1, 1]),
            }
        )

    # zwrotka
    response = {
        "parameters": parameters,
        "r_squared": utils.format_float_to_string(model.rsquared),
        "adj_r_squared": utils.format_float_to_string(model.rsquared_adj),
        "f_p_value": utils.format_float_to_string(model.f_pvalue),
        "prediction_results": None,
    }

    if test_df is not None:
        try:
            X_test = test_df[feature_cols]
            y_test = test_df[target_column]

            X_test_with_const = sm.add_constant(X_test, has_constant="add")
            y_pred = model.predict(X_test_with_const)

            response["prediction_results"] = {
                "mse": utils.format_float_to_string(mean_squared_error(y_test, y_pred)),
                "r2": utils.format_float_to_string(r2_score(y_test, y_pred)),
            }
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TEST_DATASET_BAD_STRUCTURE",
            )

    return JSONResponse(status_code=200, content=response)
