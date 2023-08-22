from __future__ import division
from builtins import range
from past.utils import old_div
from past.types import basestring
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import vg
from vg import helpers as my


def trans_prob(rain, thresh=0.0001):
    """Estimates transition probabilities.

    Parameter
    --------
    rain : 1d array
    thresh : float, optional
        threshold a value must exceed to be interpreted as rain [m]

    Returns
    -------
    trans : (2, 2) ndarray
        Transition probabilities.
        [[P(no rain -> no rain), P(no rain -> rain],
         [P(rain -> no rain),    P(rain -> rain)]]
    """
    rain_mask = rain >= thresh
    n = float(len(rain))
    diffs = np.diff(rain_mask.astype(int))
    no_change_mask = diffs == 0
    p11 = np.sum(rain_mask[:-1] & no_change_mask)
    p00 = np.sum(~rain_mask[:-1] & no_change_mask)
    p10 = np.sum(diffs == -1)
    p01 = np.sum(diffs == 1)
    return np.array([[p00, p01], [p10, p11]]) / n


def spell_lengths(rain, thresh=0.001):
    """Calculate dry and wet spell lenghts.

    Parameter
    ---------
    rain : 1d array
    thresh : float, optional
        threshold a value must exceed to be interpreted as rain [m]

    Returns
    -------
    dry : 1d int array
    wet : 1d int array
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rain_mask = rain >= thresh
    dry = np.array(
        [stop_i - start_i + 1 for start_i, stop_i in my.gaps(~rain_mask)]
    )
    wet = np.array(
        [stop_i - start_i + 1 for start_i, stop_i in my.gaps(rain_mask)]
    )
    return dry, wet


def plot_exceedance(
    obs,
    sim=None,
    kind="depth",
    thresh=0.001,
    fig=None,
    axs=None,
    figsize=None,
    lkwds=None,
    draw_scatter=None,
    *pargs,
    **pkwds,
):
    """Plot exceedance probability.

    Parameter
    --------
    obs : 1-dim ndarray
        observations
    sim : 1 or 2-dim ndarray, optional
        simulated values. If 2-dimensional, median, and range of those values
        will be plotted.
    kind : "depth", "dry", "wet" or "all", optional
        What to plot:
        - depth: rain intensity
        - dry: lengths of dry spells
        - wet: lengths of wet spells
        - all
    thresh : float, optional
        threshold a value must exceed to be interpreted as rain [m].
        only needed if kind != "depth".
    """
    if draw_scatter is None:
        draw_scatter = not bool(fig)
    if kind == "all":
        if fig is None:
            fig, axs = plt.subplots(
                ncols=3, sharey=True, figsize=figsize, constrained_layout=True
            )
        kinds = "depth", "dry", "wet"
    elif isinstance(kind, basestring):
        kinds = (kind,)
    elif fig is None or axs is None:
        fig, axs = plt.subplots(
            figsize=figsize, sharey=True, constrained_layout=True
        )
        axs = (axs,)
        kinds = (kind,)
    if lkwds is None:
        lkwds = {}

    def rel_ranks(n):
        return (np.arange(float(n)) + 0.5) / n

    def plot_sim_single(ax, sim):
        sim_conv = conv(sim)
        ranks = rel_ranks(len(sim_conv))
        sim_sorted = np.sort(sim_conv)[::-1]
        ax.loglog(sim_sorted, ranks, *pargs, **pkwds)

    edgecolor = lkwds.pop("edgecolor", "black")
    for ax_i, kind in enumerate(kinds):
        ax = axs[ax_i]

        if kind == "depth":

            def conv(x):
                valid_x = x[np.isfinite(x)]
                return valid_x[valid_x >= thresh]

            ax.set_xlabel("depth [mm]")
        elif kind == "dry":

            def conv(x):
                return spell_lengths(x, thresh)[0]

            ax.set_xlabel("dry spell length")
        elif kind == "wet":

            def conv(x):
                return spell_lengths(x, thresh)[1]

            ax.set_xlabel("wet spell length")

        if np.ndim(sim) == 1:
            plot_sim_single(ax, sim)

        if np.ndim(sim) == 2:
            sim = np.asarray(sim)
            for sim_ in sim:
                plot_sim_single(ax, sim_)

        if draw_scatter:
            obs_conv = conv(obs)
            ranks = rel_ranks(len(obs_conv))
            obs_sorted = np.sort(obs_conv)[::-1]
            ax.scatter(
                obs_sorted,
                ranks,
                marker="o",
                edgecolor=edgecolor,
                # edgecolor=(1, 0, 0, .5),
                facecolor=(0, 0, 0, 0),
                **lkwds,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(max(5e-4, ranks[0]), ranks[-1])
            # ax.set_ylim(ranks[0], ranks[-1])
            ax.grid(True)
    axs[0].set_ylabel("1 - F(X)")
    # fig.tight_layout(rect=(0, .1, 1, 1))
    return fig, axs


def richardson_model_occ(T, trans_pp):
    """Simulate rain occurrence with Richardson's (1981) model (2 state, first
    order Markov Chain).

    Parameter
    ---------
    T : int
        Length of simulated time series
    trans_pp : (2, 2) ndarray
        Transition probabilities (see trans_prob).
        [[P(no rain -> no rain), P(no rain -> rain],
         [P(rain -> no rain),    P(rain -> rain)]]

    Returns
    -------
    occs : (T,) bool ndarray
        Rain occurrence.

    References
    ----------
    Richardson, C. W. (1981). Stochastic simulation of daily precipitation,
    temperature, and solar radiation. Water Resources Research, 17(1),
    182-190. doi:10.1029/WR017i001p00182

    See also
    --------
    trans_prob : Estimates transition probabilities given a time series.
    """
    pp = trans_pp
    T = int(round(T))
    occs = np.full(T, False, dtype=bool)
    rr = vg.rng.random(T)
    # decide on the first occurrence with unconditional probability
    occs[0] = rr[0] < pp[0].sum()
    for t in range(1, T):
        occs[t] = rr[t] < (
            pp[int(occs[t - 1]), 1] / pp[:, int(occs[t - 1])].sum()
        )
    return occs


def richardson_model(T, rain, thresh=0.0001):
    if T is None:
        T = len(rain)
    trans_pp = trans_prob(rain)
    occs = richardson_model_occ(T, trans_pp)
    rain_sim = np.zeros(int(round(T)))
    rain_sim[occs] = vg.rng.choice(rain[rain > thresh], np.sum(occs))
    return rain_sim


def get_hyd_years_mask(times):
    return (times.month == 10) & (times.day == 1) & (times.hour == 0)


def hyd_year_sums(
    data_xar, fun_name="sum", full_years=True, hyd_years_mask=None
):
    time = data_xar.time
    if hyd_years_mask is None:
        hyd_years_mask = get_hyd_years_mask(time.dt)
    hyd_year = hyd_years_mask.cumsum() - 1
    hyd_year += hyd_year.time[0].dt.year
    if fun_name == "sum":
        summed = data_xar.groupby(hyd_year).sum("time").rename(group="hydyear")
    elif fun_name == "mean":
        summed = (
            data_xar.groupby(hyd_year).mean("time").rename(group="hydyear")
        )
    time_first = time[0]
    month = int(time_first.dt.month)
    day = int(time_first.dt.day)
    hyd_year_coord = pd.date_range(
        f"{summed.hydyear.values[0]}-{month}-{day}",
        f"{summed.hydyear.values[-1] + 1}-{month}-{day}",
        freq="Y",
    )
    summed = summed.assign_coords(dict(hydyear=hyd_year_coord))
    if full_years:
        if not hyd_years_mask[0]:
            summed = summed.isel(hydyear=slice(1, None))
        time_last = time[-1]
        if not ((time_last.dt.month == 9) & (time_last.dt.day == 30)):
            summed = summed.isel(hydyear=slice(None, -1))
    return summed


if __name__ == "__main__":
    import vg

    vg.conf = vg.config_konstanz
    met_vg = vg.VG(("R", "theta"))
    rain = old_div(met_vg.data_raw[0], 24)
    plot_exceedance(rain, kind="all")
    plt.show()
