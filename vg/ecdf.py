import numpy as np
from scipy import stats, interpolate
import matplotlib.pyplot as plt


class ECDF:
    def __init__(self, data, data_min=None, data_max=None):
        self.data = data
        fin_mask = np.isfinite(data)
        data_fin = data[fin_mask]
        if data_min is None:
            data_min = data_fin.min()
        if data_max is None:
            data_max = data_fin.max()
        sort_ii = np.argsort(data_fin)
        self.ranks_rel = np.full(len(data), np.nan)
        self.ranks_rel[fin_mask] = (
            stats.rankdata(data_fin, "min") - 0.5
        ) / len(data_fin)
        self._data_sort_pad = np.concatenate(
            ([data_min], data_fin[sort_ii], [data_max])
        )
        self._ranks_sort_pad = np.concatenate(
            ([0], self.ranks_rel[fin_mask][sort_ii], [1])
        )
        self._cdf = interpolate.interp1d(
            self._data_sort_pad,
            self._ranks_sort_pad,
            bounds_error=False,
            fill_value=(0, 1),
        )
        self._ppf = interpolate.interp1d(
            self._ranks_sort_pad,
            self._data_sort_pad,
            # bounds_error=False,
            fill_value=(data_min, data_max),
        )

    def cdf(self, x=None):
        if x is None:
            return self.ranks_rel
        return np.where(np.isfinite(x), self._cdf(x), np.nan)

    def ppf(self, p=None):
        if p is None:
            return self.data
        return np.where(np.isfinite(p), self._ppf(p), np.nan)

    def plot_cdf(self, fig=None, ax=None, *args, **kwds):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        ax.plot(self._data_sort_pad, self._ranks_sort_pad, *args, **kwds)
        return fig, ax
