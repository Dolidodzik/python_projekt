import io
import pandas as pd
from fastapi import HTTPException


class DataFrameUtils:

    @staticmethod
    def read_dataframe(contents):

        try:
            df = pd.read_csv(
                io.StringIO(contents.decode("utf-8")),
                sep=None,
                engine="python",
                na_values=[
                    "N/A", "NA", "NULL", "null",
                    "", "NaN", "nan", "None"
                ],
                keep_default_na=True,
                skip_blank_lines=True,
                skipinitialspace=True,
                parse_dates = True
            )

            df = df.dropna(how="all")

            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].astype(str).str.strip()
                df.loc[df[col].isin(["", "nan", "None"]), col] = pd.NA

            df = df.convert_dtypes()

            return df

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format: {str(e)}"
            )

    @staticmethod
    def parse_columns(columns, df):

        if columns is None:
            return None

        cols = [c.strip() for c in columns.split(",")]

        for col in cols:
            if col not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{col}' not found"
                )

        return cols

    @staticmethod
    def get_numeric_columns(df, columns=None):

        if columns:
            numeric_cols = [
                c for c in columns
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            raise HTTPException(
                status_code=400,
                detail="No numeric columns found"
            )

        return numeric_cols