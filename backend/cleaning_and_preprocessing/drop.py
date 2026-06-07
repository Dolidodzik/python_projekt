from fastapi import HTTPException


class ColumnDropper:

    def drop(self, df, column):

        if column is None or column.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="Column name cannot be empty"
            )

        if column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column}' not found"
            )

        return df.drop(columns=[column])


def drop_column_data(df, column):
    return ColumnDropper().drop(df, column)