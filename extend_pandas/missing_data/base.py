import pandas as pd

from tools.extend_pandas.missing_data.analysis.base import AnalysisMissingnessAdapter

try:
    del pd.DataFrame.missing
except AttributeError:
    pass


@pd.api.extensions.register_dataframe_accessor("missing")
class MissingMethods:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def analysis(self):
        return AnalysisMissingnessAdapter(self._obj)
