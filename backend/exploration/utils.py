import pandas as pd
import io
from fastapi import HTTPException, UploadFile, status
from typing import List, Optional

class FileHandler:
    """Obsługa plików CSV – odczyt i walidacja kolumn."""

    @staticmethod
    def read_csv_safe(file: UploadFile) -> pd.DataFrame:
        """Odczytuje plik CSV i zwraca DataFrame. W razie błędu rzuca HTTPException."""
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

    @staticmethod
    def validate_columns(
        df: pd.DataFrame,
        columns: Optional[str],
        numeric_only: bool = True
    ) -> List[str]:
        """
        Waliduje podane kolumny (lista rozdzielona przecinkami) i zwraca listę nazw.
        Jeśli columns = None – zwraca wszystkie odpowiednie kolumny (numeryczne lub wszystkie).
        """
        if columns is None:
            if numeric_only:
                return df.select_dtypes(include='number').columns.tolist()
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