import pandas as pd

from tools.extend_pandas.missing_data.analysis.base import AnalysisMissingnessAdapter
from tools.extend_pandas.missing_data.imputation.plots import PlotImputerAdapter

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

    @property
    def imputation(self):
        return PlotImputerAdapter(self._obj)
