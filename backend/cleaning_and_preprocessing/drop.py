from fastapi import HTTPException


def drop_column_data(df, column):
    df = df.copy()

    if column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column}' not found"
        )

    return df.drop(columns=[column]).reset_index(drop=True)