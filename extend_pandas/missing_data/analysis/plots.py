from typing import Optional, Tuple, Union

import matplotlib
import upsetplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import missingno as msno

from tools.extend_pandas.missing_data.base_adapter import MissingMethodsAdapter


class PlotMissingnessAdapter(MissingMethodsAdapter):

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
            variables = self.get_columns_with_missing_values()

        return (
            self._obj.loc[:, variables].isna()
            .value_counts()
            .pipe(lambda df: upsetplot.plot(df, **kwargs))
        )

    @staticmethod
    def add_labels(x, y):
        fixed_x_coord_to_write_label = max(y) * 1.25
        for i in range(len(y)):
            plt.text(fixed_x_coord_to_write_label, x[i], y[i], ha='center', fontsize=18)

    def mice_plot(self, missing_columns: Optional[list] = None, sorted_output_by: str = 'frequency',
                  drop_zeros: bool = True, missing_comb_exact_match: bool = True,
                  figsize: Tuple[Union[int, float]] = (6.6, 6)):

        is_missing_table, missing_freq_table = self.missing_combinations_freq_table(missing_columns,
                                                                                    sorted_output_by,
                                                                                    drop_zeros,
                                                                                    missing_comb_exact_match)
        data = is_missing_table
        table_cell_width = int(np.round(len(data.index) / len(data.columns)))
        all_columns_for_new_width = []
        for col in data.columns:
            all_columns_for_new_width.extend([col] * table_cell_width)

        data = data.loc[:, all_columns_for_new_width]

        row_labels = range(len(data.index))
        col_labels = is_missing_table.columns
        xticks = np.arange(-0.5, len(data.index), table_cell_width)
        yticks = np.arange(-0.5, len(data.index), 1)

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [10, 1]},
                                      sharey=True)
        fig.subplots_adjust(wspace=0)

        cmap = matplotlib.colors.ListedColormap(['lightblue', 'tomato'])
        bounds = [0, 0.5, 1]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(data, cmap=cmap, norm=norm)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(table_cell_width / 2 - 0.5,
                                                                            len(data.index),
                                                                            table_cell_width)))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(col_labels))
        ax.tick_params(axis='x', which='minor', labelsize=18)
        ax.yaxis.set_ticklabels([])

        ybarh = (missing_freq_table / self._obj.shape[0]).values.round(3)
        ax2.barh(row_labels, ybarh)
        self.add_labels(row_labels, ybarh)
        ax2.xaxis.set_ticklabels([])
        ax2.set_xlim([0, max(ybarh) * 1.5])
        ax2.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax2.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(table_cell_width / 2 - 0.5,
                                                                             len(data.index),
                                                                             table_cell_width)))
        ax2.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(col_labels))
        ax2.grid(visible=True, axis='y', color='k')
