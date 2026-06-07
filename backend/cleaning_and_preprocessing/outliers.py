import numpy as np
from fastapi import HTTPException
from .utils import DataFrameUtils


class OutlierRemover:

    def remove(self, df, method, columns=None, threshold=None):

        numeric_cols = DataFrameUtils.get_numeric_columns(df, columns)

        if method == "iqr":
            threshold = threshold or 1.5

            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr

                df = df[(df[col] >= lower) & (df[col] <= upper)]

        elif method == "zscore":
            threshold = threshold or 3

            for col in numeric_cols:
                z = (df[col] - df[col].mean()) / df[col].std()
                df = df[np.abs(z) < threshold]

        else:
            raise HTTPException(status_code=400, detail="Invalid method")

        return df


def remove_outliers_data(df, method, columns=None, threshold=None):
    return OutlierRemover().remove(df, method, columns, threshold)