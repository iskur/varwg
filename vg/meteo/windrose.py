#!/usr/bin/env python
"""
Provides functions to plot windroses when given wind directions.
"""
from __future__ import division
from builtins import str
from builtins import zip
from builtins import range
from past.utils import old_div
import datetime
import itertools
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------#
def _time_part_sort(datetimes, values, sub_format_str):
    """Groups "values" as a nested list according to the "sub_format_str" of
    the "datetimes"."""
    assert len(datetimes) == len(values), \
        "datetimes and values have unequal length (%d and %d)" % \
        (len(datetimes), len(values))
    time_of_values = np.array([dt.strftime(sub_format_str)
                               for dt in datetimes])
    grouped_values = []
    times = sorted(set(time_of_values))
    for time_key in times:
        grouped_values += [values[np.where(time_of_values == time_key)]]
    return times, grouped_values
#-----------------------------------------------------------------------------#
def _yscale_subplots(fig=None, per_type=False, regrid=False):
    """Sets a common y-scale to all subplots.  If per_type is set to True,
    y-scales are distinguished by the type of the subplots."""
    if fig is None:
        fig = plt.gcf()

    if per_type:
        key_func = type
    else:
        key_func = lambda x: "the one to rule them all"

    # see http://matplotlib.sourceforge.net/faq/howto_faq.html#\
    # find-all-objects-in-figure-of-a-certain-type
    ylim_getable = lambda sub: hasattr(sub, "get_ylim")
    ylim_setable = lambda sub: hasattr(sub, "set_ylim")

    # find the y-limits of each subplot
    ymins, ymaxs = {}, {}
    for subplot in fig.findobj(ylim_getable):
        ymin, ymax = subplot.get_ylim()
        sub_type = key_func(subplot)
        if sub_type not in ymins:
            ymins[sub_type], ymaxs[sub_type] = [], []
        ymins[sub_type].append(ymin)
        ymaxs[sub_type].append(ymax)

    # find the extremes for each type of subplot
    ymin, ymax = {}, {}
    for sub_type in list(ymins.keys()):
        ymin[sub_type] = min(ymins[sub_type])
        ymax[sub_type] = max(ymaxs[sub_type])


    # set ylims (and reset grids)
    for subplot in fig.findobj(ylim_setable):
        sub_type = key_func(subplot)
        subplot.set_ylim(ymin=ymin[sub_type], ymax=ymax[sub_type])
        if regrid:
            subplot.set_rgrids(np.linspace(1e-6, ymax[sub_type], 10))
#-----------------------------------------------------------------------------#
def _wind_direction_theta(classes):
    """Generates classes for directions suitable for polar axes plotting.
    Abstraction is a pain here and documenting is also hard, so sorry."""
    classes = np.where(classes > 360, classes - 360, classes)

    # avoid that a bar wraps the wrong way around the arc by cutting out the
    # overlapping class, in other words:
    # i'm a genie in a bottle, you gotta wrap me the right way
    overlap_i = int(np.argwhere(classes[:-1] - classes[1:] > 0))
    classes = np.r_[classes[:overlap_i], classes[overlap_i + 1:]]
    theta = np.pi * classes / 180
    # fill up the gap with two classes
    class_width = abs(classes[0] - classes[1])
    classes_gap = np.array([359.999 - old_div(class_width, 4),
                            360.001 + old_div(class_width, 4)])
    theta_gap = np.pi * classes_gap / 180
    return classes, theta, theta_gap, overlap_i
#-----------------------------------------------------------------------------#
def mind_the_gap(hist, overlap_i):
    hist_gap = np.r_[hist[overlap_i], hist[overlap_i]]
    hist = np.r_[hist[:overlap_i], hist[overlap_i + 1:]]
    return hist, hist_gap
#-----------------------------------------------------------------------------#
def _plot_speed_hist(speed_hist, speed_bins, bar_func, theta, theta_gap,
                     theta_width, overlap_i, *args, **kwds):
    # "transpose" the speed_hists as they can be easily plotted speed-class
    # and not sector-wise
    speed_hist = speed_hist.T
    speed_hist[np.isnan(speed_hist)] = 0
    speed_cdf = np.cumsum(speed_hist, axis=0)
    lowers = np.zeros_like(speed_cdf)
    lowers[1:] = speed_cdf[:-1]
    for ii, (lower, upper) in enumerate(zip(lowers, speed_hist)):
        hist, hist_gap = mind_the_gap(upper, overlap_i)
        l_hist, l_hist_gap = mind_the_gap(lower, overlap_i)
        label = "<= %.2f m/s" % speed_bins[ii + 1]
        bars = bar_func(theta[1:], hist, theta_width * .75, bottom=l_hist,
                        align="center", label=label, *args, **kwds)
        bars_gap = bar_func(theta_gap, hist_gap, .5 * theta_width,
                            bottom=l_hist_gap, align="center", *args,
                            **kwds)
        for bar_ in itertools.chain(bars, bars_gap):
            bar_.set_facecolor(plt.cm.jet(ii * len(speed_bins) * 10))
            bar_.set_alpha(0.5 if "alpha" not in kwds else kwds["alpha"])
    plt.legend(loc="center right", bbox_to_anchor=(1.55, 0.5),
              borderaxespad=0.)
#-----------------------------------------------------------------------------#
def _plot_dir_hist(hist, speed_hist=None, speed_bins=None, fig=None,
                   subfig=False, ticks=True, *args, **kwds):
    """Helps to combine common ground between windrose and
    _dir_hist_diff_generic.
    Can also be used to plot a windrose if the histogram is already known. But
    in this case the hist has to be passed as hist[::-1]!
    """
    # we know the wind directions have to be in the range (90,450], so we
    # do not use the classes that were returned by np.histogram
    classes = np.linspace(90, 450, len(hist) + 1)
    classes, theta, theta_gap, overlap_i = _wind_direction_theta(classes)

    try:
        overlap_i = overlap_i[0]
    except (KeyError, TypeError):
        pass
    hist, hist_gap = mind_the_gap(hist, overlap_i)

    # polar axes of subfigs are defined when the subfig is created and not
    # here.
    if not subfig:
        if fig is None:
            fig = plt.figure()
        ax = fig.add_axes([.1 if speed_hist is None else -.04, 0.1, 0.8, 0.8],
                          polar=True)
        bar_func = ax.bar
    else:
        bar_func = plt.bar
    theta_width = abs(theta[1] - theta[0])
    if speed_hist is None:
        bars = bar_func(theta[:-1], hist, theta_width * .75, bottom=0.0,
                        align='center', *args, **kwds)
        bars_gap = bar_func(theta_gap, hist_gap, 0.5 * theta_width, bottom=0.0,
                            align='center', *args, **kwds)
    else:
        _plot_speed_hist(speed_hist, speed_bins, bar_func, theta, theta_gap,
                         theta_width, overlap_i, *args, **kwds)

    if speed_hist is None:
        # set colors according to frequency/speed
        all_hists = list(itertools.chain(hist, hist_gap))
        all_bars = itertools.chain(bars, bars_gap)
        rel_hists = [old_div(hist, float(sum(all_hists))) for hist in all_hists]
        for freq, bar_ in zip(rel_hists, all_bars):
            bar_.set_facecolor(plt.cm.jet(freq * 10))
            bar_.set_alpha(0.5 if "alpha" not in kwds else kwds["alpha"])

    # we transformed the classes, so we also have to lie about the labels
    xicks = plt.xticks()[0]
    if ticks:
        xlabels = list(range(90, -1, -45)) + list(range(315, 90, -45))
        xlabels = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
#        yicks = plt.yticks()[0]
#        plt.yticks(yicks, [str(ytick) for ytick in yicks])
    else:
        xlabels = [""] * len(xicks)
    yicks = plt.yticks()[0]
    plt.yticks(yicks, [""] * len(yicks))
    plt.xticks(xicks, xlabels)
    return fig
#-----------------------------------------------------------------------------#
def seasonal_windroses(datetimes, wind_dirs, n_sectors=32,
                       time_format_str="%m", fig=None, speed=None, title=None,
                       normalize=True, separate=False, *args, **kwds):
    """Plots a figure with subplots of windroses.
    Each subplot is a windrose containing only directions from the measurement
    times signified by "time_format_str". The default ("%m") plots one windrose
    for each distinct month."""
    fig = plt.figure() if fig is None else fig
    times, sorted_values = _time_part_sort(datetimes, wind_dirs,
                                           time_format_str)
    if speed is None:
        sorted_speeds = [None] * len(times)
    else:
        times, sorted_speeds = _time_part_sort(datetimes, speed,
                                               time_format_str)

    # find out how to best fit the subplots in a wide rectangle
    fig_width = int(np.sqrt(len(times))) + 1
    fig_height = fig_width
    while fig_width * (fig_height - 1) >= len(times):
        fig_height -= 1

    # dummy to get a colorbar
    full_subplot = plt.subplot(111, polar=True)
    windrose(wind_dirs, n_sectors, fig=full_subplot, subfig=True, speed=speed,
             *args, **kwds)
    if title is not None:
        plt.suptitle(title)

    # needed to fit to a common scale
    subplots = []
    for sub_i, (time, wind_dir, speed) in \
        enumerate(zip(times, sorted_values, sorted_speeds)):
        if separate:
            windrose(wind_dir, n_sectors, speed=speed)
        else:
            subplots.append(
                plt.subplot(fig_height, fig_width, sub_i + 1, polar=True))
            windrose(wind_dir, n_sectors, fig=subplots[-1], subfig=True,
                     speed=speed, ticks=False, *args, **kwds)
        plt.title(str(time), fontsize="small")

    if normalize:
        # rescale all subfigs
        _yscale_subplots(fig, regrid=True)
        # remove all yticklabels (they are misplaced anyway)
        for subplot in subplots:
            former_labels = subplot.get_yticklabels()
            subplot.set_yticklabels([""] * len(former_labels))
    return fig
#-----------------------------------------------------------------------------#
def rel_ranks(values):
    """Returns ranks of values in the range [0,1]."""
    from scipy import stats
    return old_div((stats.stats.rankdata(values) - .5), len(values))
#-----------------------------------------------------------------------------#
def equal_num_bins(values, n_bins, round=True):
    """Returns bins so that values is cut into n_bins bins that each contain
    the same number of values' values."""
    bins = np.zeros(n_bins)
    bins[-1] = values.max()
    value_ranks = rel_ranks(values)
    for ii, perc in enumerate(np.linspace(0, 1, n_bins)[1:-1]):
        bins[ii + 1] = np.min(values[value_ranks > perc])
    return bins
#-----------------------------------------------------------------------------#
def scatter(wind_dirs, speed):
    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    wind_dirs = wind_dirs[np.isfinite(wind_dirs)]
    wind_dirs = -1 * wind_dirs + 450
    plt.scatter(wind_dirs, speed, marker="o", facecolors=(0, 0, 0, 0),
               edgecolors=(0, 0, 0, .5))
    plt.ylim(0, speed.max())
    xicks = plt.xticks()[0]
    xlabels = list(range(90, -1, -45)) + list(range(315, 90, -45))
    plt.xticks(xicks, xlabels)
    yicks = plt.yticks()[0]
    plt.yticks(yicks, [str(ytick) for ytick in yicks])
#-----------------------------------------------------------------------------#
def windrose(wind_dirs, n_sectors=32, speed=None, speed_bins=6, fig=None,
             subfig=False, ticks=True, normalize=True, *args, **kwds):
    """Plots a windrose (histogram with polar axes) of the given wind_dirs.
    If speed_bins is an int, equally filled speed_bins will be created."""
    fig = plt.figure() if fig is None else fig

    # transform directions to enable a the windrose representation
    # also works with "+ 90" but the resulting classes are less intuitive
    wind_dirs = -1 * wind_dirs + 450

    # enforce bins and ignore the classes given by np.histogram. we do not
    # want the sectors to change according to the data.
    bins = np.linspace(90 - old_div(180., n_sectors), 450 + old_div(180., n_sectors),
                       n_sectors + 2)
    hist = np.histogram(wind_dirs, bins)[0]
    hist[0] += hist[-1]
    hist = hist[:-1]
    if normalize:
        hist = old_div(hist.astype(float), np.sum(hist))
    if speed is not None:
        if type(speed_bins) == int:
            speed_bins = equal_num_bins(speed, speed_bins)
        s_hists = [np.histogram(speed[(wind_dirs > left) &
                                      (wind_dirs <= right)],
                                speed_bins)[0]
                   for left, right in zip(bins[:-1], bins[1:])]
        s_hists[0] += s_hists[-1]
        s_hists = s_hists[:-1]
        # normalize all the classes to the frequency of wind in the individual
        # sectors
        # we do not need the speeds anymore! continue with the histogram
        speed = np.array(s_hists, dtype=float)
        # normalize by hand and not with normed=True in np.histogram, because
        # "Note that the sum of the histogram values will not be equal to 1
        # unless bins of unity width are chosen; it is not a probability mass
        # function" (numpy documentation)
        speed /= speed.sum(axis=1)[:, np.newaxis]
        # make the speeds add up to the frequency of the wind falling in this
        # sector
        speed *= hist[:, np.newaxis]
    return _plot_dir_hist(hist, speed, speed_bins, fig, subfig, ticks, *args,
                          **kwds)
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    # so lets do some examples..
    # fake up wind directions following a normal distribution
    dirs = np.random.normal(loc=180, scale=45, size=365 * 4)
    # overwrite values that are not in the range (0,360]
    dirs[(dirs < 0) & (dirs >= 360)] = 0
    # generate a datetime array (one year of 6h-timesteps) to plot seasonal
    # windroses
    datetimes = np.array([datetime.datetime(2000, 1, 1) +
                          datetime.timedelta(hours=hour)
                          for hour in range(0, 365 * 4 * 6, 6)])

    # the most simple example
    windrose(dirs)
    # additional keyword arguments are passed to the plt.bar/ax.bar function.
    # here: no transparency for the bars
    windrose(dirs, alpha=1)
    # plots by month per default
    seasonal_windroses(datetimes, dirs, title="Wind directions by month")
    # plot by hour
    seasonal_windroses(datetimes, dirs, time_format_str="%H",
                       title="Wind directions by hour")

    # fake up some speed data
    weibull_ppf = lambda x, alpha, beta: (old_div(-np.log(1 - x), alpha)) ** (old_div(1, beta))
    speeds = weibull_ppf(np.random.random(365 * 4), 1, 1.5)

#    import wind_station as ws
#    konst = ws.WindStation("Messstation Langenargen")
#    seasonal_windroses(konst.datetime, konst.direction, time_format_str="%H",
#                       speed=konst.speed)

#    windrose(konst.direction)
#    windrose(konst.direction, 32, konst.speed)
#    speed_bins = np.linspace(0, konst.speed.max(), 5)
##    windrose(konst.direction[konst.speed < speed_bins[1]])
#
    plt.show()
