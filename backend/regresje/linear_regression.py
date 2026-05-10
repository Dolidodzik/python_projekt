import pandas as pd
from typing import Optional

from . import validations
from . import transformations as transformations_module
from . import ols
from .regression_model_factory import RegressionModelFactory
from . import utils
import math


def train_linear_regression(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    mode: str,
    columns: Optional[str],
    target_column: str,
    transformations: Optional[str],
    method: str = "ols",
    alpha: Optional[float] = None,
    l1_ratio: Optional[float] = None,
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

    train_all_rows_count = int(len(train_df))
    train_iqr_outliers_count = utils.iqr_outliers_count(train_df, feature_cols)
    train_vif = utils.vif_dict(train_df, feature_cols)
    train_vif_serializable = {
        k: ("+inf" if v == float("inf") else "-inf" if v == float("-inf") else v)
        for k, v in train_vif.items()
    }
    warnings = []
    if train_all_rows_count < len(feature_cols):
        warnings.append("CURSE_OF_DIMENSIONALITY")
    if len(train_vif) > 0:
        finite_vifs = [
            v for v in train_vif.values() if isinstance(v, (int, float)) and not math.isnan(v)
        ]
        max_vif = max(finite_vifs) if len(finite_vifs) > 0 else None
        if max_vif is not None and max_vif > 10:
            warnings.append("BIG_MULTICOLINEARITY")
        elif max_vif is not None and max_vif >= 5:
            warnings.append("MODERATE_MULTICOLINEARITY")
    if train_all_rows_count > 0 and (train_iqr_outliers_count / train_all_rows_count) > 0.15:
        warnings.append("HIGH_OUTLIER_RATIO")
    explanations = {
        "warnings": warnings,
        "train_iqr_outliers_count": train_iqr_outliers_count,
        "train_all_rows_count": train_all_rows_count,
        "train_vif": train_vif_serializable,
    }

    if method == "ols":
        return ols.run_ols(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_column=target_column,
            explanations=explanations,
        )

    model = RegressionModelFactory.create(
        method=method,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )
    return model.run(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_column=target_column,
        explanations=explanations,
    )
