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
    explanations: Optional[dict] = None,
):
    X = train_df[feature_cols]
    y = train_df[target_column]
    X_with_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_with_const).fit()

    conf_int = model.conf_int(alpha=0.05)

    parameters = []

    const_name = "const"
    parameters.append(
        {
            "name": "intercept",
            "coefficient": utils.format_float_to_string(model.params.get(const_name)),
            "p_value": utils.format_float_to_string(model.pvalues.get(const_name)),
            "standard_error": utils.format_float_to_string(model.bse.get(const_name)),
            "conf_int_lower": utils.format_float_to_string(
                conf_int.loc[const_name, 0] if const_name in conf_int.index else None
            ),
            "conf_int_upper": utils.format_float_to_string(
                conf_int.loc[const_name, 1] if const_name in conf_int.index else None
            ),
        }
    )

    for feat in feature_cols:
        parameters.append(
            {
                "name": feat,
                "coefficient": utils.format_float_to_string(model.params.get(feat)),
                "p_value": utils.format_float_to_string(model.pvalues.get(feat)),
                "standard_error": utils.format_float_to_string(model.bse.get(feat)),
                "conf_int_lower": utils.format_float_to_string(
                    conf_int.loc[feat, 0] if feat in conf_int.index else None
                ),
                "conf_int_upper": utils.format_float_to_string(
                    conf_int.loc[feat, 1] if feat in conf_int.index else None
                ),
            }
        )

    response = {
        "parameters": parameters,
        "r_squared": utils.format_float_to_string(model.rsquared),
        "adj_r_squared": utils.format_float_to_string(model.rsquared_adj),
        "f_p_value": utils.format_float_to_string(model.f_pvalue),
        "prediction_results": None,
        "explanations": explanations,
    }

    if explanations is not None:
        if model.rsquared < 0.2:
            explanations["warnings"].append("LOW_R2")
        if model.f_pvalue is not None and float(model.f_pvalue) > 0.05:
            explanations["warnings"].append("MODEL_NOT_STATISTICALLY_SIGNIFICANT")

    if test_df is not None:
        try:
            X_test = test_df[feature_cols]
            y_test = test_df[target_column]

            X_test_with_const = sm.add_constant(X_test, has_constant="add")
            X_test_with_const = X_test_with_const.reindex(
                columns=X_with_const.columns, fill_value=0
            )
            y_pred = model.predict(X_test_with_const)

            response["prediction_results"] = {
                "mse": utils.format_float_to_string(mean_squared_error(y_test, y_pred)),
                "r2": utils.format_float_to_string(r2_score(y_test, y_pred)),
            }
            if explanations is not None:
                test_r2 = float(r2_score(y_test, y_pred))
                if (float(model.rsquared) - test_r2) > 0.1:
                    explanations["warnings"].append("LIKELY_OVERFITTING")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TEST_DATASET_BAD_STRUCTURE",
            )

    return JSONResponse(status_code=200, content=response)
