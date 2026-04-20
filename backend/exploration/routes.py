from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd

from .utils import read_csv_safe, validate_columns, get_numeric_columns
from .core import basic_stats, column_types_report, correlation_matrix, perform_pca

router = APIRouter()

@router.post("/basic-stats")
async def basic_stats_endpoint(
    file: UploadFile = File(...),
    columns: Optional[str] = Form(None)
):
    
    #Zwraca podstawowe statystyki dla wybranych lub wszystkich kolumn numerycznych.
    
    df = read_csv_safe(file)
    col_list = validate_columns(df, columns, numeric_only=True)

    if not col_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Brak kolumn numerycznych do analizy."
        )

    try:
        stats = basic_stats(df, col_list)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas obliczeń: {str(e)}"
        )

    return JSONResponse(content=stats)

@router.post("/column-types")
async def column_types_endpoint(file: UploadFile = File(...)):
    
    #Analizuje typy kolumn w pliku CSV.
    
    df = read_csv_safe(file)
    report = column_types_report(df)
    return JSONResponse(content=report)

@router.post("/correlation")
async def correlation_endpoint(
    file: UploadFile = File(...),
    method: str = Form("pearson"),
    columns: Optional[str] = Form(None)
):
    
    #Oblicza macierz korelacji dla kolumn numerycznych.
    
    if method not in ["pearson", "spearman", "kendall"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Metoda musi być 'pearson', 'spearman' lub 'kendall'."
        )

    df = read_csv_safe(file)
    col_list = validate_columns(df, columns, numeric_only=True)

    if len(col_list) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Potrzeba co najmniej dwóch kolumn numerycznych do obliczenia korelacji."
        )

    try:
        corr = correlation_matrix(df, col_list, method)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas obliczeń: {str(e)}"
        )

    return JSONResponse(content=corr)

@router.post("/pca")
async def pca_endpoint(
    file: UploadFile = File(...),
    columns: Optional[str] = Form(None),
    n_components: Optional[int] = Form(None),
    standardize: bool = Form(True)
):
    
    #Wykonuje analizę głównych składowych.
    
    df = read_csv_safe(file)
    col_list = validate_columns(df, columns, numeric_only=True)

    if len(col_list) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PCA wymaga co najmniej dwóch kolumn numerycznych."
        )

    try:
        pca_result = perform_pca(df, col_list, n_components, standardize)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas PCA: {str(e)}"
        )

    return JSONResponse(content=pca_result)

@router.post("/describe-all")
async def describe_all_endpoint(file: UploadFile = File(...)):
    
    
    df = read_csv_safe(file)

    type_report = column_types_report(df)

    numeric_cols = get_numeric_columns(df)
    stats = basic_stats(df, numeric_cols) if numeric_cols else {}

    
    corr = None
    if len(numeric_cols) >= 2:
        corr = correlation_matrix(df, numeric_cols, method='pearson')

    report = {
        "column_types": type_report,
        "basic_statistics": stats,
        "correlation_pearson": corr
    }

    return JSONResponse(content=report)