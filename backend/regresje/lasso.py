import pandas as pd
from fastapi.responses import JSONResponse
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from typing import Optional, List

from . import utils


def run_lasso(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feature_cols: List[str],
    target_column: str,
    alpha: float,
):
    X = train_df[feature_cols]
    y = train_df[target_column]

    model = Lasso(alpha=alpha)
    model.fit(X, y)

    y_pred_train = model.predict(X)
    r2_train = r2_score(y, y_pred_train)
    adj_r2_train = utils.adjusted_r_squared_from_r2(
        r2=r2_train, n=len(train_df), p=len(feature_cols)
    )

    parameters = []
    parameters.append(
        {
            "name": "intercept",
            "coefficient": utils.format_float_to_string(float(model.intercept_)),
            "p_value": None,
            "standard_error": None,
            "conf_int_lower": None,
            "conf_int_upper": None,
        }
    )

    for feat, coef in zip(feature_cols, model.coef_):
        parameters.append(
            {
                "name": feat,
                "coefficient": utils.format_float_to_string(float(coef)),
                "p_value": None,
                "standard_error": None,
                "conf_int_lower": None,
                "conf_int_upper": None,
            }
        )

    response = {
        "parameters": parameters,
        "r_squared": utils.format_float_to_string(r2_train),
        "adj_r_squared": utils.format_float_to_string(adj_r2_train),
        "f_p_value": None,
        "prediction_results": None,
    }

    if test_df is not None:
        X_test = test_df[feature_cols]
        y_test = test_df[target_column]
        y_pred = model.predict(X_test)
        response["prediction_results"] = {
            "mse": utils.format_float_to_string(mean_squared_error(y_test, y_pred)),
            "r2": utils.format_float_to_string(r2_score(y_test, y_pred)),
        }

    return JSONResponse(status_code=200, content=response)

