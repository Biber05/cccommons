import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cccommons.plot.plotter import Plotter


class MultiPlot(Plotter):
    """"""

    def __init__(self, data: pd.DataFrame, plot_dir: str, suffix: str = None):
        super().__init__(data, plot_dir, suffix)

    def hist(self, split, x):
        """
        Args:
            split: column to separate multi plot
            x: x axis of each sub plot

        """
        grid = sns.FacetGrid(self._data, col=split)
        ax = grid.map(plt.hist, x)
        self._save(self._name("multi_hist", "{}_{}".format(split, x)), ax)
        plt.clf()
        return self

    def reg_plot(self, col, row, x, y):
        """
        Args:
            col: column to separate multi plot
            row: row to separate multi plot
            x: x axis of each sub plot
            y: y axis of each sub plot

        """
        grid = sns.FacetGrid(self._data, col=col, row=row, margin_titles=True)
        # todo - too many indices
        ax = grid.map(sns.regplot(x, y))
        self._save(self._name("multi_point", "{}_{}_{}_{}".format(col, row, x, y)), ax)
        plt.clf()
        return self

    def bar(self, col, x, y):
        """
        Args:
            col: column to separate multi plot
            x: x axis of each sub plot
            y: y axis of each sub plot
        """
        # todo - int object not iterable
        grid = sns.FacetGrid(self._data, col=col)
        ax = grid.map(sns.barplot(x, y))
        self._save(self._name("multi_bar", "{}_{}_{}".format(col, x, y)), ax)
        plt.clf()
        return self

    def scatter(self, col, x, y):
        """

        Args:
            col: column to separate multi plot
            x: x axis of each sub plot
            y: y axis of each sub plot

        """
        g = sns.FacetGrid(self._data, hue=col, height=5)
        g.add_legend()
        ax = g.map(plt.scatter, x, y)
        self._save(self._name("multi_scatter", "{}_{}_{}".format(col, x, y)), ax)
        plt.clf()
        return self

    def point(self, col, x, y):
        """

        Args:
            col: column to separate multi plot
            x: x axis of each sub plot
            y: y axis of each sub plot

        """
        g = sns.FacetGrid(self._data, hue=col, height=5)
        g.add_legend()
        ax = g.map(sns.pointplot, x, y)
        self._save(self._name("multi_point", "{}_{}_{}".format(col, x, y)), ax)
        plt.clf()
        return self
