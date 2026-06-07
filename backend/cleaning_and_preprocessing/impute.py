from sklearn.impute import KNNImputer
from fastapi import HTTPException
from .utils import DataFrameUtils


class Imputer:

    def impute(
        self,
        df,
        method,
        columns=None,
        n_neighbors=5,
        weights="uniform",
        metric="nan_euclidean"
    ):

        numeric_cols = DataFrameUtils.get_numeric_columns(df, columns)

        if not numeric_cols:
            raise HTTPException(
                status_code=400,
                detail="No numeric columns available for imputation"
            )

        if method == "median":

            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())

        elif method == "knn":

            if n_neighbors <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="n_neighbors must be > 0"
                )

            imputer = KNNImputer(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric
            )

            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid method (use 'median' or 'knn')"
            )

        categorical_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in categorical_cols:
            df[col] = df[col].fillna("Unknown")

        return df


def impute_data(
    df,
    method,
    columns=None,
    n_neighbors=5,
    weights="uniform",
    metric="nan_euclidean"
):
    return Imputer().impute(
        df,
        method,
        columns,
        n_neighbors,
        weights,
        metric
    )