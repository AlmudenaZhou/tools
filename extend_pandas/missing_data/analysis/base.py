import itertools
from typing import Optional, Union

import pandas as pd
import numpy as np
from itertools import chain, combinations

from tools.extend_pandas.missing_data.analysis.plots import PlotMissingnessAdapter


class AnalysisMissingnessAdapter(PlotMissingnessAdapter):

    def print_value_counts_per_column(self, variables: Optional[Union[list, pd.Index]] = None):
        if variables is None:
            variables = self._obj.columns
        for column in variables:
            print(f'------------ {column} ------------')
            print(self._obj[column].value_counts(dropna=False))
            print('\n')

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
            return self._filter_zero_rows_by_column(missing_var_summ, 'n_missing')
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
            pct_missing=lambda df: df["n_missing"] / self._obj.shape[1] * 100,
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

    def create_shadow_matrix(self, true_string: Union[str, bool, int, float] = "Missing",
                             false_string: Union[str, bool, int, float] = "Not Missing",
                             missing_suffix: str = '_NA', only_missing: bool = False) -> pd.DataFrame:
        """
        Creates a shadow matrix. A matrix with the same dimensions of the original df with True or False if the value
        was originally NaN.
        :param true_string: value to represents the NaN value in the shadow matrix
        :param false_string: value to represents the complete value in the shadow matrix
        :param missing_suffix: suffix added to the shadow matrix columns
        :param only_missing: if True, it only makes the shadow to the columns with NaN values
                             if False, it makes it for all the columns
        :return:
        """
        return (
            self._obj
                .isna()
                .pipe(lambda df: df[df.columns[df.any()]] if only_missing else df)
                .replace({False: false_string, True: true_string})
                .add_suffix(missing_suffix)
        )

    def bind_shadow_matrix(self, true_string: str = "Missing", false_string: str = "Not Missing",
                           missing_suffix: str = '_NA', only_missing: bool = False,
                           inplace: bool = False) -> pd.DataFrame:
        """
        Adds the shadow matrix to the original dataset
        :param true_string: value to represents the NaN value in the shadow matrix
        :param false_string:  value to represents the complete value in the shadow matrix
        :param missing_suffix: suffix added to the shadow matrix columns
        :param only_missing: if True, it only makes the shadow to the columns with NaN values
                             if False, it makes it for all the columns
        :param inplace: if True, it saves the result in the original object
        :return:
        """
        df_with_shadow_matrix = pd.concat(
            objs=[
                self._obj,
                self.create_shadow_matrix(
                    true_string=true_string,
                    false_string=false_string,
                    missing_suffix=missing_suffix,
                    only_missing=only_missing
                )
            ],
            axis="columns"
        )
        if inplace:
            self._obj = df_with_shadow_matrix
        return df_with_shadow_matrix

    def missing_scan_count_with_default_na_representation(self, elements_to_skip: list = None, show_zeros: bool = True):
        """
        Returns the number of occurrences of the default na representations. The user can choose to avoid specific
        elements from the list
        :param elements_to_skip: elements that will be skipped in the occurrence count
        :param show_zeros: if True, it keeps the columns that do not contain any possible nan. If False, it drops them.
        :return:
        """
        search = self._all_missing_value_repr
        if elements_to_skip is not None:
            search = [search_val for search_val in search if search_val not in elements_to_skip]
        return self.missing_scan_count(search, show_zeros)

    def missing_scan_count(self, search, show_zeros: bool = True) -> pd.DataFrame:
        """
        Returns the number of occurrences of the elements in search per column
        :param show_zeros: if True, it keeps the columns that do not contain any possible nan. If False, it drops them.
        :param search: list of possible missing value representations
        :return:
        """

        scan_count = (
            self._obj.apply(axis="rows", func=self._strip_cat_cols)
                .apply(axis="rows", func=lambda column: column.isin(search))
                .sum()
                .reset_index()
                .rename(columns={"index": "variable", 0: "n"})
                .assign(pct=lambda df: df['n'] / self._obj.shape[0] * 100,
                        original_type=self._obj.dtypes.reset_index()[0])
        )

        if not show_zeros:
            return self._filter_zero_rows_by_column(scan_count, 'n')
        return scan_count

    def missing_combinations_freq_table(self, missing_columns: Optional[list] = None,
                                        sorted_output_by: str = 'frequency', drop_zeros: bool = True,
                                        missing_comb_exact_match: bool = True):
        """
        Returns 2 df:
            - A df that contains a representation of each combination. It has 1 when the column is in the combination
            and 0 when it is not.
            - A table with all the different combinations of the columns that ends with missing_suffix
        and counts the number of missing values in that combination
        :param missing_columns: list of columns of which the table will be created
        :param sorted_output_by: if frequency, it sorts by the number of occurrences. If any other value, it sorts by
                                 the combination.
        :param drop_zeros: drop the columns with 0s
        :param missing_comb_exact_match: if True it only counts when the NaNs of the row are only the ones that are in
                                         the combination. If False, the combination can be a subset of the total NaNs
                                         in the row.
        :return:
        """

        if missing_columns is None:
            missing_columns = self.get_columns_with_missing_values()

        if missing_comb_exact_match:
            joint_missing_freq_table = self._obj.loc[:, missing_columns].isna().value_counts()
            joint_missing_freq_table_index = joint_missing_freq_table.index.to_frame(index=False)

            missing_freq_table_index = joint_missing_freq_table_index.apply(lambda row: tuple(row[row].index), axis=1)
            missing_freq_table = pd.Series(joint_missing_freq_table.values, index=missing_freq_table_index)

            is_missing_table = joint_missing_freq_table_index.replace({True: 1, False: 0})

            if sorted_output_by != 'frequency':
                combs = list(chain.from_iterable(combinations(missing_columns, i)
                                                 for i in range(1, len(missing_columns) + 1)))
                is_missing_table.index = missing_freq_table_index
                missing_freq_table = missing_freq_table.loc[:, combs]
                is_missing_table = is_missing_table.loc[:, combs]
                is_missing_table.reset_index(drop=True, inplace=True)

        else:
            combs = list(chain.from_iterable(combinations(missing_columns, i)
                                             for i in range(1, len(missing_columns) + 1)))
            combs.insert(0, 'no_missing')
            missing_freq_table = pd.Series(0, index=combs)
            is_missing_table = pd.DataFrame(0, index=range(len(missing_freq_table.index)), columns=missing_columns)

            for idx, comb in enumerate(combs):
                if comb != 'no_missing':
                    num_cols = len(comb)
                    missing_values_in_comb = self._obj.loc[:, comb].sum(axis=1)
                    are_all_comb_missing = missing_values_in_comb == num_cols
                    missing_freq = are_all_comb_missing.sum()
                    is_missing_table.loc[idx, comb] = 1
                    missing_freq_table[comb] = missing_freq
                else:
                    missing_freq = (self._obj.loc[:, missing_columns].sum(axis=1) == 0).sum()
                    is_missing_table.loc[idx, :] = 0
                    missing_freq_table[comb] = missing_freq
        if drop_zeros:
            no_zeros_mask = (missing_freq_table != 0).values
            missing_freq_table = missing_freq_table.loc[no_zeros_mask]
            is_missing_table = is_missing_table.loc[no_zeros_mask]
        if sorted_output_by == 'frequency':
            sorted_idx = np.argsort(missing_freq_table.values)
            is_missing_table = is_missing_table.iloc[sorted_idx]
            is_missing_table.reset_index(drop=True, inplace=True)
            missing_freq_table = missing_freq_table[sorted_idx]
        return is_missing_table, missing_freq_table
