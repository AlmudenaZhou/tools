from matplotlib import pyplot as plt
import seaborn as sns

from tools.extend_pandas.missing_data.base_adapter import MissingMethodsAdapter


class PlotImputerAdapter(MissingMethodsAdapter):

    def distribution_comparison_plot(self, imputer, column_to_compare, figsize=(16, 8), **plt_kwargs):
        _, _ = plt.subplots(figsize=figsize)

        sns.kdeplot(self._obj.SkinThickness, label="Original Distribution", **plt_kwargs)
        for model_name, model in imputer._models.items():
            data_imputed = model.transform(self._obj)
            sns.kdeplot(data_imputed[column_to_compare], label=f'model: {model_name}', **plt_kwargs)

        plt.legend()
