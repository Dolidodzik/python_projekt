import json
import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from typing import Optional, List, Tuple


def apply_transformations(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feature_cols: List[str],
    transformations: Optional[str],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    # probojemy zrobic transofmracje
    new_feature_cols = []
    if transformations:
        try:
            transformations = json.loads(transformations)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="INVALID_TRANSFORMATIONS",
            )

        for transformation in transformations:
            col = transformation.get("column")
            if (
                col not in feature_cols
            ):  # kazda transofrmacja musi miec kolumna dostepna w feature_cols
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="UNKNOWN_TRANSFORMATION_COLUMN",
                )
            transformation_type = transformation.get("type")
            new_col_name = "TRANSFORMATION_" + col + "_" + transformation_type
            if transformation_type == "log":
                if train_df[col].min() <= 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="LOG_TRANSFORMATION_NOT_ALLOWED_FOR_NON_POSITIVE_VALUES",
                    )
                train_df[new_col_name] = np.log(train_df[col])

                if test_df is not None and test_df[col].min() <= 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="LOG_TRANSFORMATION_NOT_ALLOWED_FOR_NON_POSITIVE_VALUES",
                    )
                if test_df is not None:
                    test_df[new_col_name] = np.log(test_df[col])

            elif transformation_type == "1/x":
                if (train_df[col] == 0).any():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="1/x_TRANSFORMATION_NOT_ALLOWED_FOR_NEGATIVE_VALUES",
                    )
                train_df[new_col_name] = 1 / train_df[col]

                if test_df is not None and (test_df[col] == 0).any():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="1/x_TRANSFORMATION_NOT_ALLOWED_FOR_NEGATIVE_VALUES",
                    )
                if test_df is not None:
                    test_df[new_col_name] = 1 / test_df[col]

            elif transformation_type == "square":
                train_df[new_col_name] = np.square(train_df[col])
                if test_df is not None:
                    test_df[new_col_name] = np.square(test_df[col])
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="UNKNOWN_TRANSFORMATION_TYPE",
                )
            new_feature_cols.append(new_col_name)

    feature_cols.extend(new_feature_cols)
    return train_df, test_df, feature_cols

