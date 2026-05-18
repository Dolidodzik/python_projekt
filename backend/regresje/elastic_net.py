import pandas as pd
from fastapi import HTTPException, status
from sklearn.linear_model import ElasticNet
from typing import Optional, List

from .base_regression_model import BaseRegressionModel


class ElasticNetRegressionModel(BaseRegressionModel):
    def _validate_params(self):
        if self.l1_ratio is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="L1_RATIO_REQUIRED",
            )
        if self.l1_ratio < 0 or self.l1_ratio > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="INVALID_L1_RATIO",
            )

    def _create_model(self):
        return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)


def run_elastic_net(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feature_cols: List[str],
    target_column: str,
    alpha: float,
    l1_ratio: float,
):
    return ElasticNetRegressionModel(alpha=alpha, l1_ratio=l1_ratio).run(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_column=target_column,
    )
