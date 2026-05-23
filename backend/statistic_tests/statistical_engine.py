import pandas as pd
import math

from .correlation import correlation
from .covariance import covariance
from .mann_whitney import mann_whitney_test
from .normality import normality_test
from .t_test import t_test
from .anova import anova
from .chi_square import chi_square


class StatisticalTestEngine:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def get_numeric_columns(self):
        return self.df.select_dtypes(include="number").columns.tolist()

    def get_categorical_columns(self):

        categorical = []
        cols = self.df.select_dtypes(exclude="number").columns.tolist()
        for col in cols:
            unique_count = self.df[col].dropna().nunique()
            #pomijamy kolumny typu Name, Ticket
            if 2 <= unique_count <= 20:
                categorical.append(col)

        return categorical

    def get_valid_numeric_columns(self):

        valid = []
        for col in self.get_numeric_columns():
            if "id" in col.lower():
                continue
            data = self.df[col].dropna()

            #minimum 2 wartości i nie stałe
            if len(data) >= 2 and data.nunique() > 1:
                valid.append(col)

        return valid

    def select_tests(self):
        tests = []
        numeric_cols = self.get_valid_numeric_columns()
        categorical_cols = self.get_categorical_columns()

        #Korelacja i kowariancja dla każdej pary kolumn numerycznych
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1:]:
                    tests.append(("correlation", col1, col2))
                    tests.append(("covariance", col1, col2))

        #chi kwadrat dla każdej pary kolumn kategorycznych
        if len(categorical_cols) >= 2:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i + 1:]:
                    tests.append(("chi_square", col1, col2))

        #Testy dla par kategoryczna, numeryczna
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            for cat in categorical_cols:
                for num in numeric_cols:
                    groups = self.df[cat].dropna().unique()
                    n_groups = len(groups)

                    valid = True
                    for g in groups:
                        if len(self.df[self.df[cat] == g][num].dropna()) < 2:
                            valid = False
                            break
                    if not valid:
                        continue

                    if n_groups == 2:
                        tests.append(("t_test", cat, num))
                        tests.append(("mann_whitney", cat, num))
                    elif n_groups >= 3:
                        tests.append(("anova", cat, num))


        #Test normalności dla każdej kolumny numerycznej
        for col in numeric_cols:
            data = self.df[col].dropna()
            if 3 <= len(data) <= 5000:
                tests.append(("normality", col))

        return tests


    def run_tests(self):
        tests = self.select_tests()

        for test in tests:
            try:
                test_name = test[0]

                if test_name == "correlation":
                    _, col1, col2 = test
                    result = correlation(self.df, "pearson", [col1, col2])
                    key = f"correlation_{col1}_{col2}"
                    self.results[key] = {
                        "method": "pearson",
                        "columns": [col1, col2],
                        "result": result.to_dict()
                    }

                elif test_name == "covariance":
                    _, col1, col2 = test
                    result = covariance(self.df, [col1, col2])
                    key = f"covariance_{col1}_{col2}"
                    self.results[key] = {
                        "columns": [col1, col2],
                        "result": result.to_dict()
                    }

                elif test_name == "chi_square":
                    _, col1, col2 = test
                    result = chi_square(self.df, col1, col2)
                    key = f"chi_square_{col1}_{col2}"
                    self.results[key] = {
                        "column_1": col1,
                        "column_2": col2,
                        **result
                    }

                elif test_name == "anova":
                    _, group_col, value_col = test
                    result = anova(self.df, group_col, value_col)
                    key = f"anova_{group_col}_{value_col}"
                    self.results[key] = {
                        "group_column": group_col,
                        "value_column": value_col,
                        **result
                    }

                elif test_name == "t_test":
                    _, group_col, value_col = test
                    result = t_test(self.df, group_col, value_col)
                    key = f"t_test_{group_col}_{value_col}"
                    self.results[key] = {
                        "group_column": group_col,
                        "value_column": value_col,
                        **result
                    }

                elif test_name == "mann_whitney":
                    _, group_col, value_col = test
                    result = mann_whitney_test(self.df, group_col, value_col)
                    key = f"mann_whitney_{group_col}_{value_col}"
                    self.results[key] = {
                        "group_column": group_col,
                        "value_column": value_col,
                        **result
                    }

                elif test_name == "normality":
                    _, col = test
                    result = normality_test(self.df, col)
                    key = f"normality_{col}"
                    self.results[key] = {
                        "column": col,
                        **result
                    }

            except Exception as e:
                if test[0] == "normality":
                    key = f"normality_{test[1]}"
                elif test[0] in ["correlation", "covariance", "chi_square"]:
                    key = f"{test[0]}_{test[1]}_{test[2]}"
                elif test[0] in ["anova", "t_test", "mann_whitney"]:
                    key = f"{test[0]}_{test[1]}_{test[2]}"
                else:
                    key = str(test)
                self.results[key] = {"error": str(e)}

    def clean_nan(self, obj):

        if isinstance(obj, dict):
            return {
                k: self.clean_nan(v)
                for k, v in obj.items()
            }

        elif isinstance(obj, list):
            return [
                self.clean_nan(v)
                for v in obj
            ]

        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None

        return obj

    def generate_report(self):

        cleaned_results = self.clean_nan(self.results)
        return {
            "rows": int(len(self.df)),
            "columns": self.df.columns.tolist(),
            "tests_performed": list(cleaned_results.keys()),
            "results": cleaned_results
        }

    def run(self):
        self.run_tests()
        return self.generate_report()