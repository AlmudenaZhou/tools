from datetime import datetime


class MissingMethodsAdapter:

    def __init__(self, obj):
        self._obj = obj
        self.missing_value_representation_by_type = {
            'categorical': ['-', 'na', 'n/a', 'null', 'nil', 'empty', '', '#n/d'],
            'datetime': ['1970-01-01', '1900-01-01', '2099-12-31', '2999-12-31', '9999-12-31'],
            'numerical': [0, -1, 99, 999, -99, -999]
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

    @staticmethod
    def _filter_zero_rows_by_column(df, column):
        return df.loc[df[column] != 0]

    def get_columns_with_missing_values(self):
        return self._obj.loc[:, self._obj.isna().sum(axis=0) > 0].columns

    @staticmethod
    def _strip_cat_cols(column):
        return column.str.strip() if column.dtype in ['object'] else column
