import pandas as pd
from fastapi import HTTPException, status
from typing import Optional

from . import validations
from . import transformations as transformations_module
from . import ols
from . import ridge
from . import lasso


def train_linear_regression(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    mode: str,
    columns: Optional[str],
    target_column: str,
    transformations: Optional[str],
    method: str = "ols",  # "ols", "ridge", or "lasso"
    alpha: Optional[float] = None,
):
    feature_cols, numeric_cols = validations.validate_and_select_features(
        train_df=train_df,
        test_df=test_df,
        mode=mode,
        columns=columns,
        target_column=target_column,
    )

    train_df, test_df, feature_cols = transformations_module.apply_transformations(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        transformations=transformations,
    )

    if method == "ols":
        return ols.run_ols(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_column=target_column,
        )

    if method == "ridge":
        if alpha is None or alpha <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="POSITIVE_ALPHA_REQUIRED",
            )
        return ridge.run_ridge(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_column=target_column,
            alpha=alpha,
        )

    if method == "lasso":
        if alpha is None or alpha <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="POSITIVE_ALPHA_REQUIRED",
            )
        return lasso.run_lasso(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_column=target_column,
            alpha=alpha,
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="UNKNOWN_METHOD",
    )
