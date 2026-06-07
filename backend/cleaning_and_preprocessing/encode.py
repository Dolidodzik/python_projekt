import pandas as pd
from sklearn.preprocessing import LabelEncoder
from fastapi import HTTPException


class CategoricalEncoder:

    def encode(self, df, method, columns=None):

        if columns:
            cols = columns
        else:
            cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

        if not cols:
            raise HTTPException(
                status_code=400,
                detail="No categorical columns found"
            )

        for col in cols:
            if col not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column {col} not found"
                )

        if method == "onehot":

            df[cols] = df[cols].fillna("Unknown")

            return pd.get_dummies(df, columns=cols)

        elif method == "label":

            for col in cols:
                le = LabelEncoder()

                df[col] = df[col].fillna("Unknown").astype(str)
                df[col] = le.fit_transform(df[col])

            return df

        raise HTTPException(
            status_code=400,
            detail="Invalid method (use 'onehot' or 'label')"
        )


def encode_data(df, method, columns=None):
    return CategoricalEncoder().encode(df, method, columns)