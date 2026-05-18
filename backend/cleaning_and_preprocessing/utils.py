import pandas as pd
import io
import csv
from fastapi import HTTPException


def read_dataframe(file_bytes: bytes) -> pd.DataFrame:
    try:
        content = file_bytes.decode("utf-8")

        try:
            dialect = csv.Sniffer().sniff(content[:1000])
            sep = dialect.delimiter
        except Exception:
            sep = ","

        df = pd.read_csv(io.StringIO(content), sep=sep)

        return df

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file format")


def parse_columns(columns: str, df: pd.DataFrame):
    if columns is None:
        return None

    cols = [c.strip() for c in columns.split(",") if c.strip()]

    if not cols:
        raise HTTPException(status_code=400, detail="Empty columns list")

    for col in cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column {col} not found")

    return cols


def get_numeric_columns(df: pd.DataFrame, columns=None):
    if columns:
        numeric_cols = [
            c for c in columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        raise HTTPException(
            status_code=400,
            detail="No numeric columns found"
        )

    return numeric_cols