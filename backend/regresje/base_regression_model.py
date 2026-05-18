from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from sklearn.metrics import mean_squared_error, r2_score

from . import utils


class BaseRegressionModel(ABC):
    def __init__(self, alpha: float, l1_ratio: Optional[float] = None):
        if alpha is None or alpha <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="POSITIVE_ALPHA_REQUIRED",
            )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self._validate_params()

    def _validate_params(self):
        return None

    @abstractmethod
    def _create_model(self):
        raise NotImplementedError

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        feature_cols: List[str],
        target_column: str,
        explanations: Optional[dict] = None,
    ):
        X = train_df[feature_cols]
        y = train_df[target_column]

        model = self._create_model()
        model.fit(X, y)

        y_pred_train = model.predict(X)
        r2_train = r2_score(y, y_pred_train)
        adj_r2_train = utils.adjusted_r_squared_from_r2(
            r2=r2_train, n=len(train_df), p=len(feature_cols)
        )

        parameters = [
            {
                "name": "intercept",
                "coefficient": utils.format_float_to_string(float(model.intercept_)),
                "p_value": None,
                "standard_error": None,
                "conf_int_lower": None,
                "conf_int_upper": None,
            }
        ]

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
            "explanations": explanations,
        }

        if explanations is not None:
            if r2_train < 0.2:
                explanations["warnings"].append("LOW_R2")

        if test_df is not None:
            X_test = test_df[feature_cols]
            y_test = test_df[target_column]
            y_pred = model.predict(X_test)
            test_r2 = float(r2_score(y_test, y_pred))
            response["prediction_results"] = {
                "mse": utils.format_float_to_string(mean_squared_error(y_test, y_pred)),
                "r2": utils.format_float_to_string(test_r2),
            }
            if explanations is not None:
                if (r2_train - test_r2) > 0.1:
                    explanations["warnings"].append("LIKELY_OVERFITTING")

        return JSONResponse(status_code=200, content=response)
