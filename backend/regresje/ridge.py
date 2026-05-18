import pandas as pd
from sklearn.linear_model import Ridge
from typing import Optional, List

from .base_regression_model import BaseRegressionModel


class RidgeRegressionModel(BaseRegressionModel):
    def _create_model(self):
        return Ridge(alpha=self.alpha)


def run_ridge(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feature_cols: List[str],
    target_column: str,
    alpha: float,
):
    return RidgeRegressionModel(alpha=alpha).run(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_column=target_column,
    )

