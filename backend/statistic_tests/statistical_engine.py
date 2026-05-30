import math

import pandas as pd
from scipy.stats import chi2_contingency, shapiro

from .anova import anova
from .chi_square import chi_square
from .correlation import correlation
from .covariance import covariance
from .mann_whitney import mann_whitney_test
from .normality import normality_test
from .t_test import t_test

ALPHA = 0.05
MIN_GROUP_SIZE = 2
MAX_CATEGORIES = 20
SHAPIRO_MIN = 3
SHAPIRO_MAX = 5000


class StatisticalTestEngine:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def get_numeric_columns(self):
        return self.df.select_dtypes(include="number").columns.tolist()

    def get_categorical_columns(self):
        categorical = []
        for col in self.df.select_dtypes(exclude="number").columns:
            unique_count = self.df[col].dropna().nunique()
            if 2 <= unique_count <= MAX_CATEGORIES:
                categorical.append(col)
        return categorical

    def get_valid_numeric_columns(self):
        valid = []
        for col in self.get_numeric_columns():
            if "id" in col.lower():
                continue
            data = self.df[col].dropna()
            if len(data) >= MIN_GROUP_SIZE and data.nunique() > 1:
                valid.append(col)
        return valid

    def _is_normal(self, series: pd.Series) -> bool:
        data = series.dropna()
        n = len(data)
        if n < SHAPIRO_MIN or n > SHAPIRO_MAX:
            return False
        _, p_value = shapiro(data)
        return p_value > ALPHA

    def _groups_meet_min_size(self, group_col: str, value_col: str) -> bool:
        for group in self.df[group_col].dropna().unique():
            n = len(self.df[self.df[group_col] == group][value_col].dropna())
            if n < MIN_GROUP_SIZE:
                return False
        return True

    def _two_group_test(self, group_col: str, value_col: str) -> str:
        groups = self.df[group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError("Wymagane dokładnie 2 grupy")

        normal_in_both = True
        for group in groups:
            values = self.df[self.df[group_col] == group][value_col].dropna()
            if not self._is_normal(values):
                normal_in_both = False
                break

        return "t_test" if normal_in_both else "mann_whitney"

    def _chi_square_valid(self, col1: str, col2: str) -> bool:
        table = pd.crosstab(self.df[col1], self.df[col2])
        if table.size == 0 or (table.values == 0).all():
            return False
        try:
            _, _, _, expected = chi2_contingency(table)
        except ValueError:
            return False
        low = (expected < 5).sum()
        return low / expected.size <= 0.2

    def select_tests(self):
        tests = []
        numeric_cols = self.get_valid_numeric_columns()
        categorical_cols = self.get_categorical_columns()

        for col in numeric_cols:
            n = len(self.df[col].dropna())
            if SHAPIRO_MIN <= n <= SHAPIRO_MAX:
                tests.append(("normality", col))

        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    tests.append(("correlation", col1, col2))
                    tests.append(("covariance", col1, col2))

        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i + 1 :]:
                if self._chi_square_valid(col1, col2):
                    tests.append(("chi_square", col1, col2))

        for cat in categorical_cols:
            for num in numeric_cols:
                if not self._groups_meet_min_size(cat, num):
                    continue

                n_groups = self.df[cat].dropna().nunique()
                if n_groups == 2:
                    tests.append((self._two_group_test(cat, num), cat, num))
                elif n_groups >= 3:
                    tests.append(("anova", cat, num))

        return tests

    def _test_key(self, test):
        name = test[0]
        if name == "normality":
            return f"normality_{test[1]}"
        return f"{name}_{test[1]}_{test[2]}"

    def run_tests(self):
        for test in self.select_tests():
            key = self._test_key(test)
            try:
                self.results[key] = self._run_one(test)
            except Exception as e:
                self.results[key] = {"error": str(e)}

    def _run_one(self, test):
        name = test[0]

        if name == "correlation":
            _, col1, col2 = test
            result = correlation(self.df, "pearson", [col1, col2])
            return {
                "method": "pearson",
                "columns": [col1, col2],
                "result": result.to_dict(),
            }

        if name == "covariance":
            _, col1, col2 = test
            result = covariance(self.df, [col1, col2])
            return {"columns": [col1, col2], "result": result.to_dict()}

        if name == "chi_square":
            _, col1, col2 = test
            return {"column_1": col1, "column_2": col2, **chi_square(self.df, col1, col2)}

        if name == "anova":
            _, group_col, value_col = test
            return {
                "group_column": group_col,
                "value_column": value_col,
                **anova(self.df, group_col, value_col),
            }

        if name == "t_test":
            _, group_col, value_col = test
            return {
                "group_column": group_col,
                "value_column": value_col,
                "reason": "obie grupy spełniają warunek normalności (Shapiro, p > 0.05)",
                **t_test(self.df, group_col, value_col),
            }

        if name == "mann_whitney":
            _, group_col, value_col = test
            return {
                "group_column": group_col,
                "value_column": value_col,
                "reason": "dane nie są normalne lub za mała próba do Shapiro — test nieparametryczny",
                **mann_whitney_test(self.df, group_col, value_col),
            }

        if name == "normality":
            _, col = test
            return {"column": col, **normality_test(self.df, col)}

        raise ValueError(f"Nieznany test: {name}")

    def clean_nan(self, obj):
        if isinstance(obj, dict):
            return {k: self.clean_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.clean_nan(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    def _count_significant(self, results):
        count = 0
        for entry in results.values():
            if not isinstance(entry, dict) or "error" in entry:
                continue
            p = entry.get("p_value")
            if p is not None and p < ALPHA:
                count += 1
        return count

    def generate_report(self):
        cleaned = self.clean_nan(self.results)
        return {
            "rows": int(len(self.df)),
            "columns": self.df.columns.tolist(),
            "summary": {
                "tests_count": len(cleaned),
                "significant_at_0_05": self._count_significant(cleaned),
            },
            "tests_performed": list(cleaned.keys()),
            "results": cleaned,
        }

    def run(self):
        self.run_tests()
        return self.generate_report()
