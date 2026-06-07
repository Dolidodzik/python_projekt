import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DataExplorer:

    def __init__(self, df: pd.DataFrame, cat_threshold: int = 10):
        self.df = df.copy()
        self.cat_threshold = cat_threshold
        self._detect_column_types()

    def _detect_column_types(self):
        numeric = self.df.select_dtypes(include='number').columns.tolist()
        categorical = []
        text = []
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            if self.df[col].nunique() <= self.cat_threshold:
                categorical.append(col)
            else:
                text.append(col)
        self.numeric_columns = numeric
        self.categorical_columns = categorical
        self.text_columns = text

    def column_types_report(self) -> Dict[str, Any]:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "text_columns": self.text_columns,
            "total_columns": len(self.df.columns),
            "numeric_count": len(self.numeric_columns),
            "categorical_count": len(self.categorical_columns)
        }

    def basic_stats(self, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        if columns is None:
            columns = self.numeric_columns
        else:
            columns = [c for c in columns if c in self.numeric_columns]

        result = {}
        for col in columns:
            series = self.df[col].dropna()
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

    def correlation_matrix(self, columns: Optional[List[str]] = None, method: str = 'pearson') -> Dict[str, Any]:
        if columns is None:
            columns = self.numeric_columns
        else:
            columns = [c for c in columns if c in self.numeric_columns]

        if len(columns) < 2:
            return {"error": "Potrzeba co najmniej dwóch kolumn numerycznych do korelacji."}

        corr_df = self.df[columns].corr(method=method, numeric_only=True)
        matrix = corr_df.where(pd.notnull(corr_df), None).values.tolist()
        return {
            "columns": columns,
            "correlation_matrix": matrix,
            "method": method
        }

    def pca(self, columns: Optional[List[str]] = None, n_components: Optional[int] = None, standardize: bool = True) -> Dict[str, Any]:
        if columns is None:
            columns = self.numeric_columns
        else:
            columns = [c for c in columns if c in self.numeric_columns]

        if len(columns) < 2:
            raise ValueError("PCA wymaga co najmniej dwóch kolumn numerycznych.")

        data = self.df[columns].dropna()
        if data.shape[0] < 2:
            raise ValueError("Po usunięciu braków brak wystarczającej liczby wierszy do PCA.")

        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
        else:
            data_scaled = data.values

        n_comp = n_components or min(data.shape[0], data.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(data_scaled)

        explained_variance_ratio = pca.explained_variance_ratio_.tolist()
        cumulative_variance = np.cumsum(explained_variance_ratio).tolist()

        loadings = {}
        for i in range(min(n_comp, 5)):
            component_name = f"PC{i+1}"
            loadings[component_name] = {
                columns[j]: float(pca.components_[i, j]) for j in range(len(columns))
            }

        return {
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance": cumulative_variance,
            "loadings": loadings,
            "n_components": n_comp,
            "n_samples_used": data.shape[0],
            "standardized": standardize
        }

    def value_distribution(self, column: str) -> Dict[str, Any]:
        if column not in self.df.columns:
            raise ValueError(f"Kolumna {column} nie istnieje.")

        series = self.df[column].dropna()
        if pd.api.types.is_numeric_dtype(series):
            hist, bin_edges = np.histogram(series, bins='auto')
            return {
                "column": column,
                "type": "numeric",
                "bins": bin_edges.tolist(),
                "counts": hist.tolist(),
                "statistics": {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "median": float(series.median())
                }
            }
        else:
            value_counts = series.value_counts().to_dict()
            value_counts = {str(k): v for k, v in value_counts.items()}
            return {
                "column": column,
                "type": "categorical",
                "values": value_counts,
                "unique_count": series.nunique()
            }

    def describe_all(self) -> Dict[str, Any]:
        stats = self.basic_stats(self.numeric_columns) if self.numeric_columns else {}
        corr = self.correlation_matrix() if len(self.numeric_columns) >= 2 else None
        return {
            "column_types": self.column_types_report(),
            "basic_statistics": stats,
            "correlation_pearson": corr
        }
