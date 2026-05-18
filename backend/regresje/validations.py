import pandas as pd
from fastapi import HTTPException, status
from typing import Optional, List, Tuple


def validate_and_select_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    mode: str,
    columns: Optional[str],
    target_column: str,
) -> Tuple[List[str], List[str]]:
    # żeby trenować musimy miec przynajmniej 1 predyktor i 1 target
    if train_df.empty or len(train_df.columns) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="INVALID_DATASET",
        )

    # predyktory można wybrać na 3 sposoby: all_parameters, include_parameters, exclude_parameters
    allowed_modes = {"all_parameters", "include_parameters", "exclude_parameters"}
    if mode not in allowed_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="INVALID_MODE",
        )

    # wybrana kolumna musi byc w dfie
    if target_column not in train_df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="TARGET_COLUMN_NOT_FOUND",
        )

    # jezeli test_df podany to musi miec identyczne kolumny jak train_df
    if test_df is not None:
        if list(test_df.columns) != list(train_df.columns):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TEST_DATASET_BAD_STRUCTURE",
            )

    # domyślnie wybieramy sobie wszystkie kolumny oprócz targeta jako predyktory
    all_feature_cols: List[str] = [c for c in train_df.columns if c != target_column]

    if mode == "all_parameters":
        feature_cols = all_feature_cols
    else:
        # jeżeli all_parameters nie jest true, to trzeba podac jakies kolumny (i dla include_parameters i dla exclude_parameters)
        if not columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="COLUMNS_REQUIRED",
            )

        # robimy ze stringa listę stringow podzielona przecinkiem
        requested = [c.strip() for c in columns.split(",") if c.strip()]

        # sprawdzamy czy wszystkie napewno sa w dfie
        unknown = [c for c in requested if c not in train_df.columns]
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="UNKNOWN_COLUMNS",
            )

        if mode == "include_parameters":
            # include_parameters - wybrane kolumny zostaja
            feature_cols = [c for c in requested if c != target_column]
        else:
            # exclude_parameters - wybrane kolumny sa usuwane
            feature_cols = [c for c in all_feature_cols if c not in requested]

    if not feature_cols:
        # jeżeli nie ma koncowo żadnych kolumn to da sie trenowac
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="NO_FEATURE_COLUMNS_SELECTED",
        )

    # upewniamy sie ze wszystkie kolumny sa numeryczne (cechy jakościowe powinny wcześniej zostać z-onehot-encodowane)
    numeric_cols = feature_cols + [target_column]
    non_numeric = [
        c for c in numeric_cols if not pd.api.types.is_numeric_dtype(train_df[c])
    ]
    if non_numeric:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="NON_NUMERIC_COLUMNS_NOT_ALLOWED",
        )

    # upewniamy sie ze wszystkie kolumny sa bez nanow (wtedy trenowanie sie wywala)
    if train_df[numeric_cols].isna().any().any():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="NANS_NOT_ALLOWED",
        )

    return feature_cols, numeric_cols

