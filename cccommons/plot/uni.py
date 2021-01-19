import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cccommons.plot.plotter import Plotter


class Plot(Plotter):
    """
    Plotting metrics using seaborn library and pandas
    """

    def __init__(self, data: pd.DataFrame, plot_dir: str, suffix: str = None):
        super().__init__(data, plot_dir, suffix)

    def hist(self, x, min_value=0, max_value=1000):
        """
        1D Histogram

        Args:
            x: column
            min_value:
            max_value:

        """
        df = self._extract(x=x, min_value=min_value, max_value=max_value)
        ax = sns.distplot(df)
        return self._save(self._name("hist", self._col_names(df)), ax)

    def hist2d(self, x, y, z=None, min_value=0, max_value=1000):
        """
        Args:
            x:
            y:
            z:
            min_value:
            max_value:

        """
        df = self._extract(x=x, min_value=min_value, max_value=max_value, y=y, z=z)

        if z is not None:
            ax = sns.relplot(x=x, y=y, hue=z, data=df)
        else:
            ax = sns.relplot(x=x, y=y, data=df)

        return self._save(self._name("hist2d", self._col_names(df)), ax)

    def hist2d_line(self, x, y, z=None, min_value=0, max_value=1000):
        """
        Args:
            x:
            y:
            z:
            min_value:
            max_value:

        """
        df = self._extract(x=x, min_value=min_value, max_value=max_value, y=y, z=z)

        if z is not None:
            ax = sns.relplot(x=x, y=y, hue=z, data=df, kind="line")
        else:
            ax = sns.relplot(x=x, y=y, data=df, kind="line")

        return self._save(self._name("hist2d_line", self._col_names(df)), ax)

    def hist2d_cat(self, x, y, z=None, min_value=0, max_value=1000):
        """
        Args:
            x:
            y:
            z:
            min_value:
            max_value:

        """
        df = self._extract(x=x, min_value=min_value, max_value=max_value, y=y, z=z)

        if z is not None:
            ax = sns.catplot(x=x, y=y, hue=z, data=df, kind="violin", split=False)
        else:
            ax = sns.catplot(x=x, y=y, data=df, kind="violin", split=True)

        return self._save(self._name("hist2d_cat", self._col_names(df)), ax)

    def box_plot(self, x, y, min_value=0, max_value=1000):
        """
        Args:
            x:
            y:
            min_value:
            max_value:

        """
        df = self._extract(x=x, min_value=min_value, max_value=max_value, y=y)

        ax = sns.boxplot(x=x, y=y, data=df)

        return self._save(self._name("box_plot", self._col_names(df)), ax)

    def scatter(self, x, y, min_value=0, max_value=1000):
        """
        Args:
            x:
            y:
            min_value:
            max_value:

        """
        df = self._extract(x=x, min_value=min_value, max_value=max_value, y=y)

        ax = sns.jointplot(x=x, y=y, data=df, kind="hex", color="k")

        return self._save(self._name("scatter", self._col_names(df)), ax)

    def strip(self, x, y, z=None, min_value=0, max_value=1000):
        """
        Args:
            x:
            y:
            z:
            min_value:
            max_value:

        """
        df = self._extract(x=x, min_value=min_value, max_value=max_value, y=y, z=z)

        if z is not None:
            ax = sns.stripplot(x=x, y=y, data=df, jitter=0.05, hue=z)
        else:
            ax = sns.stripplot(x=x, y=y, data=df, jitter=0.05)

        return self._save(self._name("strip", self._col_names(df)), ax)

    def heat(self, x, y, z, min_value=0, max_value=1000):
        """
        Args:
            x:
            y:
            z:
            min_value:
            max_value:

        """
        df = self._extract(x=x, min_value=min_value, max_value=max_value, y=y, z=z)

        ax = sns.heatmap(df, annot=True)

        return self._save(self._name("heat", self._col_names(df)), ax)

    def facet(self, x, y, z, min_value=0, max_value=1000):
        """
        Args:
            x:
            y:
            z:
            min_value:
            max_value:

        """
        df = self._extract(x=x, min_value=min_value, max_value=max_value, y=y, z=z)

        ax = sns.FacetGrid(df, col=x, row=y)
        # todo - takes a long time
        self._save(self._name("facetX", self._col_names(df)), ax)
        ay = ax.map(plt.hist, z)
        return self._save(self._name("facetZ", self._col_names(df)), ay)
