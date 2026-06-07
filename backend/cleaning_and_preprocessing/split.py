from sklearn.model_selection import train_test_split
from fastapi import HTTPException


class DatasetSplitter:

    def split(self, df, test_size, random_state=None, stratify=None):

        if not isinstance(test_size, float) and not isinstance(test_size, int):
            raise HTTPException(
                status_code=400,
                detail="test_size must be a number"
            )

        if not 0 < test_size < 1:
            raise HTTPException(
                status_code=400,
                detail="test_size must be between 0 and 1"
            )

        stratify_col = None

        if stratify is not None:

            if stratify not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail="Stratify column not found"
                )

            if df[stratify].isnull().any():
                raise HTTPException(
                    status_code=400,
                    detail="Stratify column contains missing values"
                )

            stratify_col = df[stratify]

        try:
            train, test = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_col
            )

        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Split error: {str(e)}"
            )

        return train, test


def split_data(df, test_size, random_state=None, stratify=None):
    return DatasetSplitter().split(df, test_size, random_state, stratify)