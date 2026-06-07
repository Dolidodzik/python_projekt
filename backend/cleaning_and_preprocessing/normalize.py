from sklearn.preprocessing import MinMaxScaler, StandardScaler
from fastapi import HTTPException
from .utils import DataFrameUtils


class Normalizer:

    def normalize(self, df, method, columns=None):

        numeric_cols = DataFrameUtils.get_numeric_columns(df, columns)

        if not numeric_cols:
            raise HTTPException(
                status_code=400,
                detail="No numeric columns to normalize"
            )

        if method == "minmax":
            scaler = MinMaxScaler()

        elif method == "zscore":
            scaler = StandardScaler()

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid method (use 'minmax' or 'zscore')"
            )

        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df


def normalize_data(df, method, columns=None):
    return Normalizer().normalize(df, method, columns)