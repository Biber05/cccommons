import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cccommons.plot.plotter import Plotter


class PairPlot(Plotter):
    def __init__(self, data: pd.DataFrame, plot_dir: str, suffix: str = None):
        super().__init__(data, plot_dir, suffix)

    def scatter(self, x, y, z):
        """

        Args:
            x: x-axis
            y: y-axis
            z: hue

        """
        g = sns.PairGrid(self._data, hue=z)
        g.map_diag(plt.hist)
        g.map_offdiag(plt.scatter)
        g.add_legend()
        self._save(self._name("pair_scatter", "{}_{}_{}".format(x, y, z)), g)
        plt.clf()
