import itertools
import pandas as pd
from datetime import datetime
import upsetplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


try:
    del pd.DataFrame.missing
except AttributeError:
    pass


@pd.api.extensions.register_dataframe_accessor("missing")
class MissingMethods:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.missing_value_representation_by_type = {
            'numerical': [0, -1, 99, 999, -99, -999],
            'categorical': ['-', 'na', 'n/a', 'null', 'nil', 'empty', '', '#n/d'],
            'datetime': ['1970-01-01', '1900-01-01', '2099-12-31', '2999-12-31', '9999-12-31']
        }
        self._all_missing_value_repr = self._calc_all_possible_na_combinations()

    def _calc_all_possible_na_combinations(self):
        string_numeric_na = [str(numeric_na) for numeric_na in self.missing_value_representation_by_type['numerical']]
        upper_string_na = [string_na.upper() for string_na in self.missing_value_representation_by_type['categorical']]
        cap_string_na = [str_na.capitalize() for str_na in self.missing_value_representation_by_type['categorical']]
        datetime_na = [datetime.strptime(dt_na, '%Y-%m-%d')
                       for dt_na in self.missing_value_representation_by_type['datetime']]
        all_original_na = [value for values in self.missing_value_representation_by_type.values() for value in values]
        return string_numeric_na + upper_string_na + cap_string_na + datetime_na + all_original_na

    def number_missing(self) -> int:
        """
        Returns the total number of NANs in the df
        :return:
        """
        return self._obj.isna().sum().sum()

    def number_complete(self) -> int:
        """
        Returns the total number of complete values in the df
        :return:
        """
        return self._obj.size - self._obj.missing.number_missing()

    def missing_variable_summary(self, show_zeros: bool = True) -> pd.DataFrame:
        """
        Returns a df that contains 3 columns:
            - variable: column names
            - n_missing: number of missing values per column
            - pct_missing: percentage of missing values per column
        :return:
        """
        missing_var_summ = self._obj.isnull().pipe(lambda df_1: (
                df_1.sum()
                    .reset_index(name="n_missing")
                    .rename(columns={"index": "variable"})
                    .assign(
                    pct_missing=lambda df_2: df_2.n_missing / len(df_1) * 100,
                )
            )
        )

        if not show_zeros:
            return missing_var_summ.loc[missing_var_summ['n_missing'] > 0]
        return missing_var_summ

    def missing_case_summary(self) -> pd.DataFrame:
        """
        Returns a df that contains 3 columns:
            - case: index
            - n_missing: number of missing values per case
            -pct_missing: percentage of missing values per case
        :return:
        """
        return self._obj.assign(
            case=lambda df: df.index,
            n_missing=lambda df: df.isna().sum(axis=1),
            pct_missing=lambda df: df["n_missing"] / df.shape[1] * 100,
        )[["case", "n_missing", "pct_missing"]]

    def missing_variable_table(self) -> pd.DataFrame:
        """
        Returns a table that contains how many columns has each number of n_missing_values in the column
        and the percentage of missing columns that those columns represents from the total.
        :return:
        """
        return (
            self._obj.missing.missing_variable_summary()
            .value_counts("n_missing")
            .reset_index()
            .rename(columns={"n_missing": "n_missing_in_variable", 0: "n_variables"})
            .assign(
                pct_variables=lambda df: df.n_variables / df.n_variables.sum() * 100
            )
            .sort_values("pct_variables", ascending=False)
        )

    def missing_case_table(self) -> pd.DataFrame():
        """
        Returns a table that contains how many rows has each number of n_missing_values in the row and
        the percentage of missing rows that those rows represents from the total.
        :return:
        """
        return (
            self._obj.missing.missing_case_summary()
            .value_counts("n_missing")
            .reset_index()
            .rename(columns={"n_missing": "n_missing_in_case", 0: "n_cases"})
            .assign(pct_case=lambda df: df.n_cases / df.n_cases.sum() * 100)
            .sort_values("pct_case", ascending=False)
        )

    def missing_variable_span(self, variable: str, span_every: int) -> pd.DataFrame:
        """
        Returns a df with missing metrics for data blocks with size span_every. It has 5 columns:
            - span_counter: number of the block
            - n_missing: number of missing values in the block
            - n_complete: number of filled values in the block
            - pct_missing: percentage of missing values in the block
            - pct_complete: percentage of filled values in the block
        :param variable: column name to apply the method
        :param span_every: window size from which calculates the missing metrics
        :return:
        """
        return (
            self._obj.assign(
                span_counter=lambda df: (
                    np.repeat(a=range(int(np.ceil(self._obj.shape[0] / span_every))), repeats=span_every)[: df.shape[0]]
                )
            )
            .groupby("span_counter")
            .aggregate(
                n_in_span=(variable, "size"),
                n_missing=(variable, lambda s: s.isnull().sum()),
            )
            .assign(
                n_complete=lambda df: df.n_in_span - df.n_missing,
                pct_missing=lambda df: df.n_missing / df.n_in_span * 100,
                pct_complete=lambda df: 100 - df.pct_missing,
            )
            .drop(columns=["n_in_span"])
            .reset_index()
        )

    def missing_variable_run(self, variable) -> pd.DataFrame:
        """
        Returns a df that analyze consecutive missing or complete values in a column:
            - run_length: number of consecutive missing or complete value
            - is_na: missing or complete
        :param variable: column name. It must be the name of the column only
        :return:
        """
        rle_list = self._obj[variable].pipe(
            lambda df: [[len(list(group)), key] for key, group in itertools.groupby(df.isnull())]
        )

        return pd.DataFrame(data=rle_list, columns=["run_length", "is_na"]).replace(
            {False: "complete", True: "missing"}
        )

    def sort_variables_by_missingness(self, ascending=False):
        """
        Sorts the columns by the number of missing values
        :param ascending:
        :return:
        """
        return self._obj.pipe(lambda df: (df[df.isna().sum().sort_values(ascending=ascending).index]))

    def create_shadow_matrix(self, true_string: str = "Missing", false_string: str = "Not Missing",
                             only_missing: bool = False,) -> pd.DataFrame:
        """
        Creates a shadow matrix. A matrix with the same dimensions of the original df with True or False if the value
        was originally NaN.
        :param true_string: value to represents the NaN value in the shadow matrix
        :param false_string: value to represents the complete value in the shadow matrix
        :param only_missing: if True, it only makes the shadow to the columns with NaN values
                             if False, it makes it for all the columns
        :return:
        """
        return (
            self._obj
                .isna()
                .pipe(lambda df: df[df.columns[df.any()]] if only_missing else df)
                .replace({False: false_string, True: true_string})
                .add_suffix("_NA")
        )

    def bind_shadow_matrix(self, true_string: str = "Missing", false_string: str = "Not Missing",
                           only_missing: bool = False) -> pd.DataFrame:
        """
        Adds the shadow matrix to the original dataset
        :param true_string: value to represents the NaN value in the shadow matrix
        :param false_string:  value to represents the complete value in the shadow matrix
        :param only_missing: if True, it only makes the shadow to the columns with NaN values
                             if False, it makes it for all the columns
        :return:
        """
        return pd.concat(
            objs=[
                self._obj,
                self._obj.missing.create_shadow_matrix(
                    true_string=true_string,
                    false_string=false_string,
                    only_missing=only_missing
                )
            ],
            axis="columns"
        )

    def missing_scan_count(self, search) -> pd.DataFrame:
        """
        Returns the number of occurrences of the elements in search per column
        :param search: list of possible missing value representations
        :return:
        """
        return (
            self._obj.apply(axis="rows", func=lambda column: column.isin(search))
            .sum()
            .reset_index()
            .rename(columns={"index": "variable", 0: "n"})
            .assign(original_type=self._obj.dtypes.reset_index()[0])
        )

    # Plotting functions ---

    def missing_variable_plot(self):
        """
        Classic line plot that shows the number of missing values in the x axis and the columns in the y axis
        :return:
        """
        df = self._obj.missing.missing_variable_summary().sort_values("n_missing")

        plot_range = range(1, len(df.index) + 1)

        plt.hlines(y=plot_range, xmin=0, xmax=df.n_missing, color="black")

        plt.plot(df.n_missing, plot_range, "o", color="black")

        plt.yticks(plot_range, df.variable)

        plt.grid(axis="y")

        plt.xlabel("Number missing")
        plt.ylabel("Variable")

    def missing_case_plot(self, bins=None):
        """
        Histogram of the missing values per row
        :return:
        """
        df = self._obj.missing.missing_case_summary()

        sns.displot(data=df, x="n_missing", binwidth=1, color="black", bins=bins)

        plt.grid(axis="x")
        plt.xlabel("Number of missings in case")
        plt.ylabel("Number of cases")

    def missing_variable_span_plot(
            self, variable: str, span_every: int, rot: int = 0, figsize=None
    ):
        """
        Plots a bar plot showing the number of complete and missing value per data block
        :param variable: column name
        :param span_every: size of each data block
        :param rot: number of degrees to rotate the x axis labels
        :param figsize: size of the plot
        :return:
        """

        (
            self._obj.missing.missing_variable_span(
                variable=variable, span_every=span_every
            ).plot.bar(
                x="span_counter",
                y=["pct_missing", "pct_complete"],
                stacked=True,
                width=1,
                color=["black", "lightgray"],
                rot=rot,
                figsize=figsize,
            )
        )

        plt.xlabel("Span number")
        plt.ylabel("Percentage missing")
        plt.legend(["Missing", "Present"])
        plt.title(
            f"Percentage of missing values\nOver a repeating span of {span_every} ",
            loc="left",
        )
        plt.grid(False)
        plt.margins(0)
        plt.tight_layout(pad=0)

    def missing_upsetplot(self, variables: list[str] = None, **kwargs):
        """
        Upset plot that shows the missing values relationships between columns
        :param variables: list of
        :param kwargs: possible variables to use in the upsetplot see
        https://upsetplot.readthedocs.io/en/stable/api.html#plotting
        :return:
        """

        if variables is None:
            variables = self._obj.columns.tolist()

        return (
            self._obj.isna()
            .value_counts(variables)
            .pipe(lambda df: upsetplot.plot(df, **kwargs))
        )
