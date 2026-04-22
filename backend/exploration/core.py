import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def basic_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, Any]]:
    
# Oblicza podstawowe statystyki dla podanych kolumn numerycznych.
    
    result = {}
    for col in columns:
        series = df[col].dropna()
        if len(series) == 0:
            result[col] = {"error": "Brak danych w kolumnie"}
            continue

        stats = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else None,
            "min": float(series.min()),
            "max": float(series.max()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
        }
        result[col] = stats
    return result

def column_types_report(df: pd.DataFrame) -> Dict[str, Any]:

#   Generuje raport o typach kolumn.
    
    numeric = df.select_dtypes(include='number').columns.tolist()
    categorical = []
    text = []

    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() <= 10:
            categorical.append(col)
        else:
            text.append(col)

    return {
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "text_columns": text,
        "total_columns": len(df.columns),
        "numeric_count": len(numeric),
        "categorical_count": len(categorical)
    }

def correlation_matrix(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'pearson'
) -> Dict[str, Any]:

    #Oblicza macierz korelacji dla podanych kolumn.

    if method not in ['pearson', 'spearman', 'kendall']:
        method = 'pearson'

    corr_df = df[columns].corr(method=method, numeric_only=True)
    # Zastąpienie NaN na None, aby JSON był poprawny
    matrix = corr_df.where(pd.notnull(corr_df), None).values.tolist()

    return {
        "columns": columns,
        "correlation_matrix": matrix,
        "method": method
    }

def perform_pca(
    df: pd.DataFrame,
    columns: List[str],
    n_components: Optional[int] = None,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    Wykonuje PCA na wybranych kolumnach numerycznych.
    """
    data = df[columns].copy()

    # Usuwamy wiersze z jakimikolwiek brakami
    data = data.dropna()
    if data.shape[0] < 2:
        raise ValueError("Za mało danych do PCA (potrzeba co najmniej 2 wiersze bez braków).")

    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.values

    if n_components is None:
        n_components = min(data.shape[0], data.shape[1])
    else:
        n_components = min(n_components, data.shape[0], data.shape[1])

    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_.tolist()
    cumulative_variance = np.cumsum(explained_variance_ratio).tolist()

    # Ładunki czynnikowe dla pierwszych 5 komponentów 
    loadings = {}
    for i in range(min(n_components, 5)):
        component_name = f"PC{i+1}"
        loadings[component_name] = {
            columns[j]: float(pca.components_[i, j])
            for j in range(len(columns))
        }

    return {
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "loadings": loadings,
        "n_components": n_components,
        "n_samples_used": data.shape[0],
        "standardized": standardize
    }