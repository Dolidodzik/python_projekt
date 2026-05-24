from sklearn.model_selection import train_test_split
from fastapi import HTTPException


def split_data(df, test_size, random_state=None, stratify=None):
    df = df.copy()

    if df.empty:
        raise HTTPException(status_code=400, detail="Empty dataset")

    if not 0 < test_size < 1:
        raise HTTPException(
            status_code=400, detail="test_size must be between 0 and 1"
        )

    stratify_col = None
    if stratify:
        if stratify not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Stratify column not found"
            )

        df = df.dropna(subset=[stratify])

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Dataset became empty after removing NaN values"
            )

        stratify_col = df[stratify]

    try:
        train, test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=stratify_col
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return train.reset_index(drop=True), test.reset_index(drop=True)