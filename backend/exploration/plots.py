from pathlib import Path
from typing import Dict, Any, List, Optional, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .core import DataExplorer


class DiagramGenerator:

    ID_COLUMNS: ClassVar[tuple] = ("passengerid", "id", "index")
    PREFERRED_NUMERIC: ClassVar[tuple] = ("Age", "Fare", "price", "value")

    def __init__(
        self,
        explorer: DataExplorer,
        output_dir: Path,
        dataset_name: str = "dane",
    ):
        self.explorer = explorer
        self.output_dir = self._ensure_dir(output_dir)
        self.dataset_name = dataset_name
        self.generated_files: Dict[str, str] = {}

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        output_dir: Path,
        dataset_name: str = "dane",
        cat_threshold: int = 10,
    ) -> "DiagramGenerator":
        df = pd.read_csv(csv_path)
        explorer = DataExplorer(df, cat_threshold=cat_threshold)
        return cls(explorer, output_dir, dataset_name)

    def _ensure_dir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _output_path(self, suffix: str) -> Path:
        return self.output_dir / f"{self.dataset_name}_{suffix}.png"

    def _analysis_columns(self) -> List[str]:
        numeric = self.explorer.numeric_columns
        filtered = [c for c in numeric if c.lower() not in self.ID_COLUMNS]
        return filtered if len(filtered) >= 2 else numeric

    def _stats_columns(self) -> List[str]:
        numeric = self.explorer.numeric_columns
        filtered = [c for c in numeric if c.lower() not in self.ID_COLUMNS]
        return filtered or numeric

    def _preferred_numeric_column(self) -> Optional[str]:
        numeric = self.explorer.numeric_columns
        for name in self.PREFERRED_NUMERIC:
            if name in numeric:
                return name
        for col in numeric:
            if col.lower() not in self.ID_COLUMNS:
                return col
        return numeric[0] if numeric else None

    def plot_correlation_heatmap(
        self,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> Path:
        columns = columns or self._analysis_columns()
        result = self.explorer.correlation_matrix(columns, method)
        if "error" in result:
            raise ValueError(result["error"])

        cols = result["columns"]
        matrix = np.array(result["correlation_matrix"], dtype=float)
        output_path = self._output_path(f"korelacja")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            xticklabels=cols,
            yticklabels=cols,
            ax=ax,
        )
        ax.set_title(f"Macierz korelacji ({method})")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    def plot_numeric_distribution(self, column: str) -> Path:
        result = self.explorer.value_distribution(column)
        if result["type"] != "numeric":
            raise ValueError(f"Kolumna {column} nie jest numeryczna.")

        bins = result["bins"]
        counts = result["counts"]
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
        output_path = self._output_path(f"rozkład_{column}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(bin_centers, counts, width=(bins[1] - bins[0]) * 0.9, edgecolor="black", alpha=0.75)
        ax.axvline(result["statistics"]["mean"], color="red", linestyle="--", label="Średnia")
        ax.axvline(result["statistics"]["median"], color="green", linestyle="--", label="Mediana")
        ax.set_xlabel(column)
        ax.set_ylabel("Liczba obserwacji")
        ax.set_title(f"Rozkład kolumny: {column}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    def plot_categorical_distribution(self, column: str, top_n: int = 10) -> Path:
        result = self.explorer.value_distribution(column)
        if result["type"] != "categorical":
            raise ValueError(f"Kolumna {column} nie jest kategoryczna.")

        items = sorted(result["values"].items(), key=lambda x: x[1], reverse=True)[:top_n]
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        output_path = self._output_path(f"rozkład_{column}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, values, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_xlabel(column)
        ax.set_ylabel("Liczba wystąpień")
        ax.set_title(f"Rozkład kategoryczny: {column}")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    def plot_pca_variance(self, columns: Optional[List[str]] = None) -> Path:
        columns = columns or self._analysis_columns()
        result = self.explorer.pca(columns)
        ratios = result["explained_variance_ratio"]
        cumulative = result["cumulative_variance"]
        labels = [f"PC{i + 1}" for i in range(len(ratios))]
        output_path = self._output_path("pca")

        fig, ax1 = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))

        ax1.bar(x, ratios, color="cornflowerblue", label="Wariancja wyjaśniona")
        ax1.set_ylabel("Udział wariancji")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_title("PCA – wyjaśniona wariancja")

        ax2 = ax1.twinx()
        ax2.plot(x, cumulative, color="darkorange", marker="o", linewidth=2, label="Skumulowana")
        ax2.set_ylabel("Wariancja skumulowana")
        ax2.set_ylim(0, 1.05)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    def plot_basic_stats_summary(self, columns: Optional[List[str]] = None) -> Path:
        columns = columns or self._stats_columns()
        stats = self.explorer.basic_stats(columns)
        if not stats:
            raise ValueError("Brak kolumn numerycznych do wizualizacji.")

        rows = []
        for col, values in stats.items():
            if "error" in values:
                continue
            rows.append({
                "kolumna": col,
                "średnia": values["mean"],
                "mediana": values["median"],
                "odch. std.": values["std"] or 0,
            })

        df = pd.DataFrame(rows).set_index("kolumna")
        output_path = self._output_path("statystyki")

        fig, ax = plt.subplots(figsize=(9, 5))
        df.plot(kind="bar", ax=ax, rot=0)
        ax.set_title("Podstawowe statystyki kolumn numerycznych")
        ax.set_ylabel("Wartość")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    def generate_all(self) -> Dict[str, Any]:
        self.generated_files = {}
        analysis_cols = self._analysis_columns()

        if len(analysis_cols) >= 2:
            path = self.plot_correlation_heatmap(columns=analysis_cols)
            self.generated_files["korelacja"] = str(path)

            path = self.plot_pca_variance(columns=analysis_cols)
            self.generated_files["pca"] = str(path)

        if self.explorer.numeric_columns:
            path = self.plot_basic_stats_summary()
            self.generated_files["statystyki"] = str(path)

            col = self._preferred_numeric_column()
            if col:
                path = self.plot_numeric_distribution(col)
                self.generated_files["rozkład_numeryczny"] = str(path)

        if self.explorer.categorical_columns:
            col = self.explorer.categorical_columns[0]
            path = self.plot_categorical_distribution(col)
            self.generated_files["rozkład_kategoryczny"] = str(path)

        return {
            "column_types": self.explorer.column_types_report(),
            "files": self.generated_files,
        }
