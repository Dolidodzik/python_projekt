import pandas as pd
import io
from fastapi import HTTPException, UploadFile, status
from typing import List, Optional

def read_csv_safe(file: UploadFile) -> pd.DataFrame:
    
    #odczyt pliku CSV
    
    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        return df
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Plik nie jest poprawnym plikiem tekstowym UTF-8."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Błąd odczytu pliku CSV: {str(e)}"
        )
    finally:
        file.file.close()

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    
    #Zwraca listę kolumn numerycznych w DataFrame.
    
    return df.select_dtypes(include='number').columns.tolist()

def get_categorical_columns(df: pd.DataFrame, threshold: int = 10) -> List[str]:
    
    #Zwraca listę kolumn kategorycznych (object/category z małą liczbą unikalnych wartości).
    categorical = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() <= threshold:
            categorical.append(col)
    return categorical

def validate_columns(
    df: pd.DataFrame,
    columns: Optional[str],
    numeric_only: bool = True
) -> List[str]:
    """
    Waliduje podane kolumny i zwraca listę nazw.
    - Jeśli columns=None, zwraca wszystkie odpowiednie (numeryczne lub wszystkie).
    - Sprawdza istnienie kolumn i opcjonalnie ich numeryczność.
    """
    if columns is None:
        if numeric_only:
            return get_numeric_columns(df)
        else:
            return df.columns.tolist()

    col_list = [c.strip() for c in columns.split(',') if c.strip()]
    if not col_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parametr 'columns' nie może być pusty."
        )

    missing = [c for c in col_list if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Kolumny nie istnieją w pliku: {missing}"
        )

    if numeric_only:
        non_numeric = [
            c for c in col_list
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Kolumny nie są numeryczne: {non_numeric}"
            )

    return col_list