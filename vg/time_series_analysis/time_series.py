import collections

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from vg import helpers as my
from vg.smoothing import smooth


def time_part_average(
    values, times, time_format_str="%w", expand=True, return_times=False
):
    """Return averages per time part, e.g. monthly means."""
    time_parts = times.time_part(times, time_format_str)
    time_parts_diffs = time_parts[1:] - time_parts[:-1]
    breaks = np.where(time_parts_diffs != 0)[0] + 1
    values_splitted = np.split(values, breaks)
    values_means = np.array([val.mean() for val in values_splitted])
    if expand:
        repeats = [len(val) for val in values_splitted]
        time_part_average.repeats = repeats
        values_means = np.repeat(values_means, repeats)
    if return_times:
        return values_means, np.r_[[times[0]], times[breaks]]
    return values_means


def matr_img(
    array,
    title=None,
    xlabels=None,
    ylabels=None,
    fig=None,
    figsize=None,
    fontsize=12,
    colorbar=False,
    text=True,
    *args,
    **kwds,
):
    fig = plt.figure(figsize=figsize) if fig is None else fig
    plt.imshow(array, interpolation="nearest", **kwds)
    if xlabels is not None:
        plt.xticks(np.arange(len(xlabels)), xlabels, rotation=45)
    if ylabels is not None:
        plt.yticks(np.arange(len(ylabels)), ylabels)
    if title is not None:
        plt.title(title)
        try:
            fig.canvas.set_window_title(
                "%s (%d)" % (title, fig.canvas.manager.num)
            )
        except AttributeError:
            pass
    ax = plt.gca()
    ax.tick_params(left="off", bottom="off", top="off", right="off")

    if colorbar:
        plt.colorbar()
    if text:
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                # do not let yourself be fooled by the strange
                # imshow-coordinate system
                plt.text(
                    y,
                    x,
                    "%.2f" % array[x, y],
                    fontsize=fontsize,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
    return fig, ax


def corr_img(data, k=0, title=None, var_names=None, rank=False, *args, **kwds):
    """Plots the correlation matrix of data using imshow"""
    if rank:
        corr = cross_rank_corr(data, k)
    else:
        corr = cross_corr(data, k)
    return matr_img(
        corr,
        title,
        xlabels=var_names,
        ylabels=var_names,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        *args,
        **kwds,
    )


def auto_cov_(data, k):
    """Return the autocovariance-vector for lag k. Variables are assumed to
    be stored in rows, with time extending across the columns."""
    data = np.asarray(data)
    if np.ndim(data) == 1:
        data = data[None, :]
    n_vars = data.shape[0]
    k = np.atleast_1d(np.abs(k))
    finite_ii = np.isfinite(data)
    n_data = finite_ii.sum(axis=1)[None, :]
    means = np.nanmean(data, axis=1)[:, None]
    auto = np.empty((len(k), n_vars))
    for ii, lag in enumerate(k):
        if lag == 0:
            auto0 = [
                np.nanmean((row - mean) ** 2) for row, mean in zip(data, means)
            ]
            auto[ii] = np.array(auto0)
        else:
            for jj, dat in enumerate(data):
                dat1, dat2 = dat[:-lag], dat[lag:]
                lag_finite_ii = np.isfinite(dat1) & np.isfinite(dat2)
                nans = np.sum(~lag_finite_ii)
                auto[ii, jj] = np.mean(
                    (dat1[lag_finite_ii] - means)
                    * (dat2[lag_finite_ii] - means)
                )
                auto[ii, jj] *= (n_data[jj] - lag - nans) / (
                    n_data[jj]
                ).astype(float)
    return np.squeeze(auto)


def auto_cov(data, k):
    """Return the autocovariance-vector for lag k. Variables are assumed to
    be stored in rows, with time extending across the columns.

    This version uses masked arrays, but is not faster than auto_cov_, which
    has a deeper nesting for loop.
    """
    data = np.copy(data)
    if np.ndim(data) == 1:
        data = data[None, :]
    n_vars = data.shape[0]
    finite_ii = np.isfinite(data)
    n_data = finite_ii.sum(axis=1)[:, None]
    data = np.ma.array(data, mask=~finite_ii)
    data -= data.mean(axis=1)[:, None]
    k = np.atleast_1d(np.abs(k))
    auto = np.empty((len(k), n_vars), dtype=float)
    for ii, lag in enumerate(k):
        if lag == 0:
            auto0 = [(row**2).mean() for row in data]
            auto[ii] = np.array(auto0)
        else:
            data1, data2 = data[:, :-lag], data[:, lag:]
            auto[ii] = (data1 * data2).mean(axis=1)
            auto[ii] *= np.ravel(((n_data - lag) / (n_data).astype(float)))
    return np.squeeze(auto)


def auto_corr(data, k):
    """Return the autocorrelation-vector for lag k.

    Parameters
    ----------
    data : (K, T) array
        K number of variables. T time steps.
    k : int or iterable
        time-lag or time lags if k is an iterable
    """
    if not isinstance(k, collections.abc.Iterable):
        k = (k,)
    return np.squeeze([auto_cov(data, lag) / auto_cov(data, 0) for lag in k])


def _partial_autocorr_uni(data, k):
    """Univariate implementation. Call partial_autocorr for multi- and
    univariate cases instead.
    Autocorrelation with the influence of the time steps between t and t-k
    removed. See Hartung p.677f. Tested according to example on p.694"""
    k = np.atleast_1d(k)
    autos = np.empty(len(k))
    r_k = auto_corr(data, 1)
    for lag_i, lag in enumerate(k):
        if lag == 0:
            autos[lag_i] = 1
        elif lag == 1:
            autos[lag_i] = r_k
        else:
            # the following block was written for k as an integer. optimization
            # in terms of not executing stuff twice is probably possible
            # read P_k-1
            P_km1 = np.atleast_1d(auto_corr(data, list(range(1, lag))))
            P_km1_rev = P_km1[::-1]
            # read P_k-1,k-1
            P_km1km1 = np.ones((lag - 1, lag - 1))
            for ii in range(lag - 1):
                if ii > 0:
                    P_km1km1[ii, :ii] = P_km1_rev[-ii:]
                if ii < lag - 2:
                    P_km1km1[ii, ii + 1 :] = P_km1[ii:-1]
            P_km1km1_inv = np.linalg.inv(P_km1km1)
            autos[lag_i] = (
                r_k - P_km1 @ P_km1km1_inv @ P_km1_rev.T
            ) / np.sqrt(
                (1 - P_km1 @ P_km1km1_inv @ P_km1.T)
                @ (1 - P_km1_rev @ P_km1km1_inv @ P_km1_rev.T)
            )
    return np.squeeze(autos)


def partial_autocorr(data, k):
    """Autocorrelation with the influence of the time steps between t and t-k
    removed. See Hartung p.677f. Tested according to example on p.694"""
    if np.ndim(data) == 1:
        return _partial_autocorr_uni(data, k)
    else:
        return np.array([_partial_autocorr_uni(var, k) for var in data])


def plot_auto_corr(
    data,
    k_range=7 * 24,
    title="",
    var_names=None,
    n_per_day=1,
    partial=False,
    n_difference=0,
    figsize=None,
    window_title=True,
    fig=None,
    ax=None,
    *args,
    **kwds,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.canvas.mpl_connect("draw_event", my.scale_yticks)
    lags = np.arange(k_range)
    corr_func = partial_autocorr if partial else auto_corr
    if isinstance(data, np.ndarray):
        d_data = np.copy(data)
        if np.ndim(data) == 1:
            d_data = d_data[np.newaxis, :]
        if np.ndim(data) == 2:
            d_data = d_data[np.newaxis, ...]
    else:
        d_data = data
    linestyles = ("-", "--", ":")
    symbols = "x^*"
    for data_2d, linestyle, symbol in zip(d_data, linestyles, symbols):
        for diff in range(n_difference + 1):
            fig.add_subplot(n_difference + 1, 1, diff + 1)
            autos = corr_func(data_2d, lags)
            for auto, color in zip(autos.T, "bgrcmykw"):
                ax.plot(
                    auto,
                    color + linestyle + symbol,
                    markerfacecolor=(0, 0, 0, 0),
                    markeredgecolor=color,
                )
            xes = list(range(0, k_range, n_per_day))
            ax.set_xticks(xes, minor=xes)
            if var_names is not None:
                ax.legend(
                    var_names,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    frameon=False,
                    borderaxespad=0.0,
                )
    ax.grid()
    fig.suptitle(title)
    if window_title:
        fig.canvas.set_window_title(
            "%s (%d)" % (title, fig.canvas.manager.num)
        )
    return fig, ax


# def cross_cov_nan(data, k):
#    """Biased cross covariance of an array containig nans."""
#    n_vars = data.shape[0]
#    means = my.nanavg(data, axis=1)[:, np.newaxis]
#    cross = np.empty((n_vars, n_vars))
#    for ii in range(n_vars):
#        for jj in range(n_vars):
#            cross[ii, jj] = my.nanavg((data[ii, :-k
def cross_cov(data, k, means=None):
    """Return the cross-covariance matrix for lag k. Variables are assumed to
    be stored in rows, with time extending across the columns."""
    #    if k == 0:
    #        # very funny, jackass
    #        return np.cov(data)
    n_vars, T = data.shape
    k_right = -abs(k) if k else None
    if means is None:
        means = np.nanmean(data, axis=1).reshape((n_vars, 1))
        ddof = 1
    else:
        ddof = 0
    cross = np.empty((n_vars, n_vars))
    for ii in range(n_vars):
        for jj in range(n_vars):
            cross[ii, jj] = np.nansum(
                (data[ii, :k_right] - means[ii]) * (data[jj, k:] - means[jj])
            ) / (T - ddof)
            # cross[ii, jj] = \
            #     np.nanmean((data[ii, :k_right] - means[ii]) *
            #                (data[jj, k:] - means[jj]), ddof=ddof)
    return cross


def cross_corr(data, k):
    """Return the cross-correlation-coefficient matrix for lag k.

    Parameters
    ----------
    data : (K, T) array
        K number of variables. T time steps.
    k : int or iterable
        time-lag or time lags if k is an iterable
    """
    if not isinstance(k, collections.abc.Iterable):
        k = (k,)
    finite_ii = np.isfinite(data)
    stds = [np.nanstd(row[row_ii]) for row, row_ii in zip(data, finite_ii)]
    stds = np.array(stds)[:, np.newaxis]
    stds_dot = stds * stds.T  # dyadic product of row-vector stds
    return np.squeeze([cross_cov(data, lag) / stds_dot for lag in k])


def plot_cross_corr(
    data,
    var_names=None,
    max_lags=10,
    figsize=None,
    fig=None,
    axss=None,
    *args,
    **kwds,
):
    K = data.shape[0]
    if var_names is None:
        var_names = np.arange(K).astype(str)
    lags = np.arange(max_lags)
    # shape: (max_lags, K, K)
    cross_corrs = np.array([cross_corr(data, k) for k in lags])
    if fig is None and axss is None:
        size = {}
        if figsize:
            size["figsize"] = figsize
        fig, axss = plt.subplots(
            K, squeeze=True, sharey=True, sharex=True, **size
        )
    for var_i in range(K):
        lines = []
        for var_j in range(K):
            # want to set the same colors as before when called with a given
            # fig and axss
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            color = colors[var_j % len(colors)]
            lines += axss[var_i].plot(
                lags, cross_corrs[:, var_i, var_j], color=color, *args, **kwds
            )
        axss[var_i].set_title(var_names[var_i])
        axss[var_i].grid(True)
        axss[var_i].set_ylabel("correlation")
    axss[-1].set_xlabel("time lag")
    plt.subplots_adjust(right=0.75, hspace=0.25)
    if fig is not None:
        fig.legend(lines, var_names, loc="center right")
    return fig, axss


def plot_scaling(
    data,
    agg_funcs=(np.mean, np.std, sp.stats.skew),
    var_names=None,
    max_agg_len=365,
    fig=None,
    axs=None,
    *args,
    **kwds,
):
    if var_names is None:
        var_names = np.arange(data.shape[0]).astype(str)
    if fig is None and axs is None:
        fig, axs = plt.subplots(len(var_names), len(agg_funcs), sharex=True)

    # should be of shape:
    # (len(agg_funcs), max_agg_len - 2, len(var_names))
    data_agg = np.array(
        [
            [
                agg_func(
                    [
                        smooth(sub, agg_len, window_function="flat")
                        for sub in data
                    ],
                    # my.sumup(data, agg_len, mean=True),
                    axis=1,
                )
                for agg_len in range(2, max_agg_len)
            ]
            for agg_func in agg_funcs
        ]
    )

    for var_i, var_name in enumerate(var_names):
        for agg_j, agg_func in enumerate(agg_funcs):
            ax = axs[var_i, agg_j]
            ax.semilogx(data_agg[agg_j, :, var_i], *args, **kwds)
            ax.grid(True)
            if var_i == 0:
                ax.set_title(agg_func.__name__)
            if agg_j == 0:
                ax.set_ylabel(var_name)
    return fig, axs


def partial_corr_(x, y, u, rank_cor=True):
    """Is the correlation between x and y because of u?.
    Note that this is only a reasonable measure if x, y and u are normally
    distributed."""
    if rank_cor:
        corr_func = my.kendalls_tau
    else:

        def corr_func(x, y):
            return np.corrcoef(x, y)[0, 1]

    r_xy = corr_func(x, y)
    r_xu = corr_func(x, u)
    r_yu = corr_func(y, u)
    return (r_xy - r_xu * r_yu) / ((1 - r_xu**2) * (1 - r_yu**2)) ** 0.5


def rank_corr_ij(data):
    """Computes the rank correlation coefficient matrix."""
    n_vars = data.shape[0]
    r_ij = np.empty((n_vars, n_vars))
    # fill the lower triangular matrix
    for ii in range(n_vars):
        for jj in range(ii):
            r_ij[ii, jj] = my.kendalls_tau(data[ii], data[jj])
        # do not compute the correlation of a variable with itself
        r_ij[ii, ii] = 1
    # fill the upper triangular matrix, exploiting symmetry
    for ii in range(n_vars):
        if ii + 1 < n_vars:
            for jj in range(ii + 1, n_vars):
                r_ij[ii, jj] = r_ij[jj, ii]
    return r_ij


def cross_rank_corr(data, k):
    """Computes the cross rank correlation for lag k."""
    if k == 0:
        # this should be much faster because it avoids doing double work
        return rank_corr_ij(data)
    n_vars = data.shape[0]
    r_ij = np.empty((n_vars, n_vars))
    for ii in range(n_vars):
        for jj in range(n_vars):
            r_ij[ii, jj] = my.kendalls_tau(data[ii, k:], data[jj, :-k])
    return r_ij


def partial_corr(data, index, diff=False, rank_cor=True, r_ij=None):
    """Returns a matrix with partial (rank) correlation coefficients of all
    variables with respect to the variables in row "index".
    If diff is given as True, the difference to the correlation coefficients
    will be returned.
    If rank_cor is given as False, the ordinary partial correlations are used.
    """
    if r_ij is None:
        if rank_cor:
            r_ij = rank_corr_ij(data)
        else:
            r_ij = cross_corr(data, 0)
    # correlation coefficients between all variables and the one under study
    r_ik = r_ij[index]
    # the same as a column array
    r_kj = r_ik[:, np.newaxis]
    # to iterate is human and to recurse is divine, but to broadcast is to
    # enter Nirwana
    partial = (r_ij - r_ik * r_kj) / ((1 - r_ik**2) * (1 - r_kj**2)) ** 0.5
    if diff:
        partial = r_ij - partial
    return partial


def hurst_coefficient(values):
    """Estimates the Hurst coefficient.

    Parameters
    ----------
    values :    1-dim ndarray
                Data representing a time-series.

    References
    ----------
    http://en.wikipedia.org/wiki/Hurst_coefficient#Estimating_the_exponent
    """
    # find n_partitions, so that the minimum number of observations in a
    # sub-time-series is 8
    N = values.size
    n_partitions = int(np.log2(float(N) / 8))
    Rn_Sn = np.zeros(n_partitions)
    region_sizes = np.zeros(n_partitions)
    for part_i in range(n_partitions):
        # form sub-time-series by reshaping, so we can do all calculation with
        # the help of axis=1
        n_parts = 2**part_i
        cutoff = -(N % n_parts)
        if cutoff == 0:
            cutoff = None
        Xt = values[:cutoff].reshape((n_parts, -1))
        Yt = Xt - Xt.mean()
        Zt = np.cumsum(Yt, axis=1)
        Rn = np.max(Zt, axis=1) - np.min(Zt, axis=1)
        # Sn = np.sqrt(np.mean((Xt - Xt.mean(axis=1)[:, np.newaxis]) ** 2))
        Sn = np.std(Xt, axis=1)
        Rn_Sn[part_i] = np.mean(Rn / Sn)
        region_sizes[part_i] = Xt.shape[1]

    # do a regression on x=log(n), y=log(Rn/Sn)
    return sp.stats.linregress(np.log2(region_sizes), np.log2(Rn_Sn))[0]


def mann_kendall(values, sign_niv=0.025):
    """Test on the significance of a trend.

    Parameters
    ----------
    values :    1-dim ndarray
                Data representing a time-series.
    sign_niv : niveau for the two-sided test

    Returns
    -------
    trend : {1, -1, 0}
        1: significant increasing trend, -1: significant decreasing trend,
        0: no significant trend

    References
    ----------
    http://www.bodensee-hochwasser.info/pdf/langzeitverhalten-bodensee-wasserstaende.pdf
    page 36f
    """
    n = len(values)
    Q = 0
    for i in range(0, n - 1):
        Q = Q + np.sum(np.sign(values[i + 1 :] - values[i]))
    sigma = ((n * (n - 1) * (2 * n + 5)) / 18.0) ** 0.5
    z = Q / sigma
    if z < sp.stats.norm.ppf(sign_niv / 2.0):
        trend = -1  # decreasing trend
    elif z > sp.stats.norm.ppf(1 - sign_niv / 2.0):
        trend = 1  # increasing trend
    else:
        trend = 0  # no significant trend
    return trend


# def test_partial_corr(data, index, alpha=.05, rank_cor=True, r_ij=None):
#    """Returns an array with True/False values whether the Null Hypothesis
#    that x and y are partially uncorrelated can be rejected.
#    See Hartung p.562"""
#    n = data.shape[1]
#    if r_ij is None:
#        r_ij = rank_corr_ij(data)
#    # to only test for partially uncorrelatedness where it makes sense, we
#    # limit ourselves to elements that are rank-correlated at all
#    # see Hartung p.560
#    corr_test_statistic = n * (n - 1) / 2. * r_ij
#    # i have no theoretical values of the test statistic, but the following
#    # transformation puts it into the standard normal world.
#    corr_test_statistic /= (n * (n - 1) * (2. * n + 5) / 18) ** .5
#    corr_mask = (np.abs(corr_test_statistic) >
#                 sp.stats.norm.ppf(1 - .5 * alpha))
#    # approximation of the theoretical K-statistic
#    partial = partial_corr(data, index, rank_cor=rank_cor, r_ij=r_ij)
#    test_statistic = np.abs(partial * (n - 3) ** .5 /
#                            (1 - partial ** 2) ** .5)
# #    return corr_mask & (test_statistic > stats.t.ppf(1 - .5 * alpha, n - 3))
#    return test_statistic > sp.stats.t.ppf(1 - .5 * alpha, n - 3)


if __name__ == "__main__":
    import vg

    met_vg = vg.VG(("theta", "ILWR", "Qsw", "rh", "u", "v"))
    met_vg.fit(2)
    simt, sim = met_vg.simulate()
    # met_vg.plot_autocorr()
    met_vg.plot_doy_scatter("theta")
