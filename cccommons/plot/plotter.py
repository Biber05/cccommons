import os
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plotter(object):
    def __init__(self, data: pd.DataFrame, plot_dir: str, suffix: str = None):
        """

        Args:
            data: pandas DataFrame as input values
        """
        self._plot_dir = plot_dir
        self._suffix = suffix
        self._data = data
        self.c = data.columns

        print("Column: {}".format(self._col_names(self._data)))

    @staticmethod
    def _col_names(df: Union[pd.DataFrame, pd.Series]) -> str:
        """

        Args:
            df: DataFrame

        Returns:
            all column names as concat string
        """
        if isinstance(df, pd.DataFrame):
            return ", ".join(map(lambda x: str(x), df.columns))
        else:
            return str(df.name)

    def _extract(
        self, x, y=None, z=None, min_value: int = 0, max_value: int = 1000
    ) -> pd.DataFrame:
        """
        Args:
            x: first column
            y: second column (optional)
            z: third column (optional) - should be categorical
            min_value: start index (default = 0)
            max_value: end index (default = 1_000)

        Returns:
            DataFrame with specified columns and rows
        """
        if x not in self.c:
            raise BaseException(
                "Column x: {} not found. Available: {}".format(
                    x, self._col_names(self._data)
                )
            )

        if y is not None and y not in self.c:
            raise BaseException(
                "Column y: {} not found. Available: {}".format(
                    y, self._col_names(self._data)
                )
            )

        if z is not None and z not in self.c:
            raise BaseException(
                "Column z: {} not found. Available: {}".format(
                    z, self._col_names(self._data)
                )
            )

        if z is not None and self._data[z].dtype not in pd.CategoricalDtype:
            raise BaseException(
                "Column z: {} should be a categorical variable but it is {}".format(
                    z, self._data[z].dtype
                )
            )

        if z is not None:
            df = self._data[[x, y, z]]
            return df.loc[min_value:max_value]
        elif y is not None:
            df = self._data[[x, y]]
            return df.loc[min_value:max_value]
        else:
            df = self._data[x]
            return df.loc[min_value:max_value]

    @staticmethod
    def release():
        """
        Close matplotlib plot viewer
        """
        plt.clf()

    def _save(self, filename, plot):
        """
        Saves image to /SHARE/plot folder
        """
        if (
            isinstance(plot, sns.FacetGrid)
            or isinstance(plot, sns.JointGrid)
            or isinstance(plot, sns.PairGrid)
        ):
            plot.savefig(os.path.join(self._plot_dir, filename))
        else:
            fig = plot.get_figure()
            fig.savefig(os.path.join(self._plot_dir, filename))
        plt.clf()
        return self

    def _name(self, func: str, param) -> str:
        if self._suffix is None:
            return "{}_{}.png".format(func, str(param))
        else:
            return "{}_{}_{}.png".format(func, str(param), self._suffix)

    @staticmethod
    def show():
        """
        Show plot with matplotlib

        Notes:
            use release() method to clear cache of plot
        """
        plt.show()
