"""Just some functions I find myself writing or searching for again and
again..."""

# The imports are not all done here.  Importing this module to use a few
# specific functions would mean pulling a lot of other imports in that are not
# needed.  As importing some of the modules can be quite costly speed-wise,
# the clarity of having the import statements at the beginning is consciously
# compromised.
from __future__ import division, print_function, with_statement

import contextlib
import inspect
import itertools
import os
import re
import sys
import warnings

from builtins import map, object, range, str, zip
from future import standard_library

standard_library.install_aliases()

from future.moves.collections import UserDict
from past.builtins import basestring
from past.utils import old_div


try:
    # should make multiprocessing less stressfull
    import dill as pickle
except ImportError:
    import pickle

import hashlib
import functools
import random
import numpy as np

try:
    from multiprocessing import cpu_count
    import numexpr as ne

    ne.set_num_threads(min(64, cpu_count()))
    NE = True
except ImportError:
    NE = False
from scipy import optimize, stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

PY2 = sys.version_info.major == 2

# Generic


def asscalar(func):
    """Return the result as a scalar if it has len == 1."""

    @functools.wraps(func)
    def wrapped(*args, **kwds):
        result = np.atleast_1d(func(*args, **kwds))
        return result.item() if result.size == 1 else result

    return wrapped


def _build_arg_str(function, *func_args, **func_kwds):
    """Builds a string that completely describes the parameters in a function
    call. This helps to cache function results that have no side-effects."""
    # remember that default arguments arrive empty in **func_kwds.
    # build dictionary of argument-names to default argument values
    spec_arg_names = inspect.getargspec(function).args
    if spec_arg_names[0] == "self":
        spec_arg_names = spec_arg_names[1:]
    arg_dict = dict.fromkeys(spec_arg_names, None)
    spec_arg_values = inspect.getargspec(function).defaults
    # spec_arg_values apply to the last n spec_arg_names
    if spec_arg_names and spec_arg_values:
        for name, val in zip(spec_arg_names[::-1], spec_arg_values[::-1]):
            arg_dict[name] = val

    # change the default values according to the passed values
    for arg, arg_name in zip(func_args, spec_arg_names):
        arg_dict[arg_name] = arg
    arg_dict.update(func_kwds)

    return "__".join(
        "%s_%s" % (key, arg_dict[key]) for key in sorted(arg_dict.keys())
    )


def pickle_cache(filepath_template="%s.pkl", clear_cache=False, warn=True):
    """Use this as a function decorator to cache the result of a function as a
    pickle-file. The filename is determined by the arguments in the function.
    If the filename turns out to be too long, a hash of it is used."""

    def function_wrapper(function):
        @functools.wraps(function)
        def pickle_function(*args, **kwds):
            # set a filename based on the arguments passed
            # arg_dict_str = sanitize(_build_arg_str(function, *args, **kwds))
            arg_dict_str = _build_arg_str(function, *args, **kwds)
            if len(arg_dict_str) > 0:
                # if isinstance(filepath_template, pathlib.Path):
                #     filepath_template = str(filepath_template)
                filepath = filepath_template % arg_dict_str
            else:
                filepath = filepath_template

            def hash_(filepath):
                name_hash = hashlib.md5(
                    os.path.basename(filepath.encode())
                ).hexdigest()
                return os.path.join(os.path.dirname(filepath), name_hash)

            filepath_hash = hash_(filepath)
            if clear_cache:
                if os.path.exists(filepath):
                    os.remove(filepath)
                if os.path.exists(filepath_hash):
                    os.remove(filepath_hash)

            if not re.match(r".*py[2,3]\.pkl$", filepath):
                # keep different pickles for different python versions
                version_str = "_%s." % ("py2" if PY2 else "py3")
                filepath = version_str.join(filepath.rsplit("."))
                # name_parts = filepath.rsplit(".")
                # filepath = "".join((name_parts[0],
                #                     "_",
                #                     "py2" if PY2 else "py3",
                #                     name_parts[1]))

            def read(filepath):
                with open(filepath, "rb") as pi_file:
                    return pickle.load(pi_file)

            if os.path.exists(filepath) or os.path.exists(hash_(filepath)):
                try:
                    if warn:
                        warnings.warn(
                            "I am not executing %s, but restoring the "
                            % repr(function)
                            + "result from its last execution with the "
                            + "same parameters. \nThe result is stored "
                            + "here: %s" % filepath
                        )
                    return read(filepath)
                except IOError:
                    if os.path.exists(filepath_hash):
                        return read(filepath_hash)
                except UnicodeDecodeError:
                    # this is probably a py2/3 incompatibility problem
                    os.remove(filepath)

            result = function(*args, **kwds)

            def dump(filepath):
                with open(filepath, "wb") as pi_file:
                    pickle.dump(result, pi_file)

            try:
                dump(filepath)
            except IOError:
                # maybe the filename was too long
                dump(hash_(filepath))
            return result

        return pickle_function

    return function_wrapper


class ADict(UserDict):
    def __add__(self, other):
        # we need a copy to work with
        left_dict = dict(self)
        left_dict.update(other)
        # make sure we can do this operation also with the returned
        # object
        return ADict(left_dict)

    def __sub__(self, other):
        left_dict = dict(self)
        if isinstance(other, dict):
            del_keys = list(other.keys())
        elif isinstance(other, basestring):
            del_keys = (other,)
        else:
            del_keys = other
        for del_key in del_keys:
            del left_dict[del_key]
        return ADict(left_dict)


def cache(*names, **name_values):
    """Use as a decorator, to supply *names attributes that can be used as
    a cache. The attributes are set to their default/None during
    compile time. The wrapped function also has a 'clear_cache'-method
    to delete those variables.

    Parameter
    ---------
    *names : str

    """

    def wrapper(function):
        @functools.wraps(function)
        def cache_holder(*args, **kwds):
            return function(*args, **kwds)

        cache_holder._cache_names = names
        cache_holder._cache_name_values = name_values
        cache_holder.clear_cache = lambda: clear_def_cache(cache_holder)
        cache_holder.clear_cache()
        return cache_holder

    return wrapper


def clear_def_cache(function, cache_names=None, cache_name_values=None):
    """I often use a simplified function cache in the form of
    'function.attribute = value'.  This function helps cleaning it up,
    i.e. setting them to None.

    Parameter
    ---------
    function : object with settable attributes
    cache_names : sequence of str or None, optional
        if None, function should have an attribute called _cache_names with
        names of attributes that are cached.
    """
    if cache_names is None:
        cache_names = function._cache_names
    if cache_name_values is None:
        cache_name_values = function._cache_name_values
    for name in cache_names:
        setattr(function, name, None)
    for name, value in list(cache_name_values.items()):
        setattr(function, name, value)


# Filesystem


@contextlib.contextmanager
def chdir(dirname):
    """Temporarily change the working directory with a with-statement."""
    old_dir = os.path.abspath(os.path.curdir)
    if dirname:  # could be an empty string
        os.chdir(dirname)
    yield
    os.chdir(old_dir)


# Numeric


def kendalls_tau(x, y):
    """Kendall's Rank correlation coefficient. See Hartung p.599f

    Examples
    --------
    >>> A = [8, 6, 5, 3.5, 1, 2, 3.5, 7]
    >>> B = [6, 7.5, 4, 1, 2, 3, 5, 7.5]
    >>> kendalls_tau(A, B)
    0.5714285714285714
    """
    assert len(x) == len(y)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    # TODO: we have to be pessimistic when coming across equal values
    y_xrank_sorted = y[np.argsort(x)]
    y_ranks = stats.rankdata(y_xrank_sorted)
    # i forcefully put in "<" instead of "<=" because i am annoyed that the
    # correlation between a variable and itself is not 1 when there are
    # equal values inside
    # comment on the comment: changed that back to do it "by the book"
    q_i = np.array(
        [np.sum(y_ranks[ii + 1 :] <= y_ranks[ii]) for ii in range(n)],
        dtype=float,
    )
    return 1 - 4 * np.sum(q_i) / (n * (n - 1))


def chi2_test(x, y, k=None, n_parameters=0):
    """Chi-square test for inequality.
    H0: x and y were sampled from the same distribution.

    Parameters
    ----------
    k : int
        Number of classes (bins)

    Returns
    -------
    p_value : float
    """
    n = len(x)
    if k is None:
        k = int(n**0.5)
        # k = n_parameters + 2
    observed, bins = np.histogram(x, k)[:2]
    expected = np.histogram(y, bins)[0]
    chi_test = np.sum(
        old_div((observed.astype(float) - expected) ** 2, expected)
    )
    # degrees of freedom:
    dof = k - n_parameters - 1
    print(chi_test, stats.chi2.ppf(0.95, dof))
    return stats.chisqprob(chi_test, dof)


def rel_ranks(values, method="average"):
    """Returns ranks of values in the range [0,1]."""
    if isinstance(values, int):
        N = values
        return (np.arange(N) + 0.5) / N
    return (stats.stats.rankdata(values, method) - 0.5) / len(values)


def val2ind(values, value):
    """Return the index of the nearest neighbor of value in values."""
    # the int-conversion is necessary.  without it, the index comes out as
    # 'numpy.int64' (on my machine), which causes "illegal subscript type"
    # errors when used as an index on netcdf-arrays.
    flat_index = int(np.argmin(np.abs(values - value)))
    if np.ndim(values) == 1:
        return flat_index
    else:
        return np.unravel_index(flat_index, values.shape)


def round_to_float(values, precision):
    """Round to nearest precision.

    >>> round_to_float([8, 12], 5.)
    array([ 10.,  10.])
    """
    values = np.asarray(values, dtype=float)
    rest = values % precision
    return np.where(
        rest > old_div(precision, 2.0),
        values + (precision - rest),
        values - rest,
    )


def fourier_approx(data, order=4, size=None, how="longest"):
    """Approximate data with a Fourier transform, using the order number of
    frequencies with the highest amplitudes.

    Parameters
    ----------
    data : 1-dim ndarray
    order : int, optional
        Number of frequencies to account for.
    size : int, optional
        Desired length of the output. If None, it will be the same as data.
    how : "longest" or "strongest", optional
    """
    if size is None:
        size = len(data)
    data_freq = np.fft.fft(data)
    if how == "longest":
        ii_below = list(range(order + 1, len(data)))
    elif how == "strongest":
        # find the order biggest amplitudes
        ii_below = np.argsort(np.abs(data_freq))[: len(data_freq) - order - 1]
    pars = np.copy(data_freq)
    pars[ii_below] = 0
    return np.fft.irfft(pars, size)


def periodic_pad(values):
    half = len(values) // 2
    return np.concatenate((values[-half:], values, values[:half]))


def interp_nan(values, times=None, max_interp=None, pad_periodic=False):
    """Remove nans from values by linear interpolation.

    Parameters
    ----------
    values : 2d array
    times : 1d array, optional
    max_interp : int
        Maximum number of subsequent nans to interpolate over.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0., np.nan, 1., np.nan, np.nan, 4.])
    >>> interp_nan(a)
    array([ 0. ,  0.5,  1. ,  2. ,  3. ,  4. ])
    >>> interp_nan(a, max_interp=1)
    array([ 0. ,  0.5,  1. ,  nan,  nan,  4. ])
    >>> a = np.arange(6, dtype=float).reshape((2, 3))
    >>> a[0, 1] = np.nan
    >>> interp_nan(a)
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]])
    """
    if pad_periodic:
        half = len(values) // 2
        values = periodic_pad(values)
    values = np.atleast_2d(np.copy(values))
    for row_i, row in enumerate(values):
        nans = np.isnan(row)
        if times is None:
            times = np.arange(values.shape[1])

        if max_interp:
            nan_beginnings = np.where(np.diff(nans.astype(int)) == 1)[0] + 1
            nan_endings = np.where(np.diff(nans.astype(int)) == -1)[0] + 1
            if nans[0]:
                nan_beginnings = np.concatenate(([0], nan_beginnings))
            if nans[-1]:
                nan_endings = np.concatenate((nan_endings, [len(nans) - 1]))

            nan_lengths = nan_endings - nan_beginnings
            for episode_i, nan_length in enumerate(nan_lengths):
                if nan_length > max_interp:
                    start_i = nan_beginnings[episode_i]
                    nans[start_i : start_i + nan_length] = False

        if np.any(nans):
            values[row_i, nans] = np.interp(
                times[nans], times[~nans], row[~nans]
            )
    if pad_periodic:
        values = values[:, half:-half]
    return np.squeeze(values)


def sumup(
    values,
    width=24,
    times_=None,
    drop_extra=True,
    mean=False,
    middle_time=True,
    sum_to_nan=False,
    acceptable_nans=6,
    max_interp=3,
):
    """Sum up width number of values along the rows. If there are surplus
    entries, they are dropped as if they were hot (Snoop Dog et al).

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10.).reshape((2, 5))
    >>> a
    array([[ 0.,  1.,  2.,  3.,  4.],
           [ 5.,  6.,  7.,  8.,  9.]])
    >>> sumup(a, 2)
    array([[  1.,   5.],
           [ 11.,  15.]])
    >>> sumup(a, 2, drop_extra=False)
    array([[  1.,   5.,   8.],
           [ 11.,  15.,  18.]])
    >>> sumup(a.ravel(), 5)
    array([ 10.,  35.])
    >>> a[0, 0] = np.nan
    >>> sumup(a, 2, mean=True)
    array([[ 1. ,  2.5],
           [ 5.5,  7.5]])
    """
    width = int(width)
    if max_interp > 0 and not sum_to_nan:
        values = interp_nan(values, max_interp=max_interp)
    if len(values.shape) == 1:
        values = values[np.newaxis, :]
    # we hack the values into a (x, width) shape, sum along the rows and
    # reshape it back
    orig_rows, orig_columns = values.shape
    surplus_columns = orig_columns % width
    if drop_extra and surplus_columns:
        values = values[:, :-surplus_columns]
        orig_columns -= surplus_columns
    elif (not drop_extra) and surplus_columns:
        last_values_mean = values[:, np.newaxis, -surplus_columns:].mean(
            axis=2
        )
        last_values_mean = np.array(last_values_mean, dtype=values.dtype)
        values = np.concatenate((values, last_values_mean), axis=1)
    values = values.reshape((values.size // width, width))
    summed_values = np.nansum(values, axis=1)

    if np.sum(np.isnan(values)) > 0:
        nan_counts = np.sum(np.isnan(values), axis=1)
        if sum_to_nan:
            summed_values[nan_counts > 0] = np.nan
        else:
            nan_ii = (nan_counts > 0) & (nan_counts <= acceptable_nans)
            summed_values[nan_ii] *= old_div(
                width, (float(width) - nan_counts[nan_ii])
            )
            summed_values[nan_counts > acceptable_nans] = np.nan
    if mean:
        summed_values = summed_values.astype(float)
        summed_values /= width - np.sum(np.isnan(values), axis=1)

    new_columns = int(np.ceil(old_div(float(orig_columns), width)))
    summed_values = summed_values.reshape((orig_rows, new_columns))

    if times_ is not None:
        if middle_time:
            # use the time in the middle between the data points
            time_shift = round(old_div(width, 2.0))
        else:
            time_shift = None
        times_ = times_[time_shift::width][: summed_values.shape[1]]
        return np.squeeze(summed_values), times_
    else:
        return np.squeeze(summed_values)


def gaps(data):
    """Return indices referring to start and end points of gaps (marked by nans
    in the given array 'data'

    Parameters
    ----------
    data : 1dim ndarray, dtype float or bool

    >>> import numpy as np
    >>> a = np.arange(20.)
    >>> a[-1] = np.nan
    >>> gaps(a)
    array([[19, 19]])
    >>> a = np.arange(20.)
    >>> a[[0, 3, 4, 5, 12, 13, -1]] = np.nan
    >>> gaps(a)
    array([[ 0,  0],
           [ 3,  5],
           [12, 13],
           [19, 19]])
    >>> a = np.arange(20.)
    >>> a[[0, 1, 3, 4, 5, 12, -2, -1]] = np.nan
    >>> gaps(a)
    array([[ 0,  1],
           [ 3,  5],
           [12, 12],
           [18, 19]])
    """
    if data.dtype == bool:
        mask = data
    else:
        mask = np.isnan(data)
    if np.all(~mask):
        return []
    if np.all(mask):
        return [[0, len(mask) - 1]]
    diff = np.diff(mask.astype(int))
    begin_ii = (np.where(diff == 1)[0] + 1).tolist()
    end_ii = np.where(diff == -1)[0].tolist()
    if mask[0]:
        if len(begin_ii) == 0:
            begin_ii = [0]
        if not end_ii[0] < begin_ii[0]:
            end_ii = [0] + end_ii
        begin_ii = [0] + begin_ii
    if (begin_ii[-1] and not end_ii) or begin_ii[-1] > end_ii[-1]:
        end_ii += [len(mask) - 1]

    return np.array([begin_ii, end_ii]).T


def list_transpose(list_):
    """Transposes a "2-dim" nested list.

    Examples
    --------
    >>> list_transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    """
    return list(map(list, list(zip(*list_))))


def csv2list(
    filename,
    startfrom=None,
    delimiter=None,
    column_ids=None,
    conversions=None,
    comment="#",
):
    """Returns a list of each column of a csv-file."""
    if conversions is None:
        # no conversion corresponds to a string-conversion.  itertools.repeat
        # gives us those for as many columns that might be there.
        conversions = itertools.repeat(str)
    if (len(np.atleast_1d(column_ids)) > 1) and (
        len(np.atleast_1d(conversions)) == 1
    ):
        # as convenience, this expands the conversions to the length of the
        # given column_ids
        column_conversions = conversions
        conversions = itertools.repeat(conversions)

    with open(filename) as csv_file:
        all_data = [
            [
                conversion(value.strip())
                for value, conversion in zip(
                    line.split(delimiter), conversions
                )
            ]
            for line in itertools.islice(csv_file, startfrom)
            if not line.lstrip().startswith(comment)
        ]

    # "transpose" rows to columns
    columns = list_transpose(all_data)

    if column_ids is None:
        return columns
    elif len(column_ids) == 1:
        # do not return a nested list if there is only one column
        return [
            column_conversions[0](value) for value in columns[column_ids[0]]
        ]
    else:
        return [
            [conversion(value) for value in columns[ii]]
            for ii, conversion in zip(column_ids, column_conversions)
        ]


def csv2dict(filename, *args, **kwds):
    """Returns a dictionary containing the columns of a csv-file.  The keys
    are taken from the first row of the file."""
    aslist = csv2list(filename, *args, **kwds)
    return {column[0].strip(): column[1:] for column in aslist}


# Plotting


class LegendSubtitleHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        # this dedents the label
        handlebox.set_width(0)
        # a dummy with zero width and no visible edge
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        patch = mpatches.Rectangle(
            [x0, y0],
            0,
            handlebox.height,
            edgecolor=(0, 0, 0, 0),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


legend_subtitle = LegendSubtitleHandler()


def square_subplots(n_variables, *args, **kwds):
    """Similar to plt.subplots, but shares x-axes column-wise and y-axes
    row - wise. Main diagonal subplots only share x-axes."""
    fig = plt.figure(*args, **kwds)
    axes = np.empty((n_variables, n_variables), dtype=object)
    for ii in range(n_variables):
        for jj in range(n_variables):
            if ii == 0:
                sharex_ax = None
            else:
                sharex_ax = axes[0, jj]

            if (jj == 0) or (ii == jj):
                sharey_ax = None
            elif ii == 0:
                sharey_ax = axes[ii, 1]
            else:
                sharey_ax = axes[ii, 0]

            axes[ii, jj] = fig.add_subplot(
                n_variables,
                n_variables,
                ii * n_variables + jj + 1,
                sharex=sharex_ax,
                sharey=sharey_ax,
            )
    return fig, axes


def splom(
    data,
    variable_names=None,
    f_kwds=None,
    h_kwds=None,
    s_kwds=None,
    opacity=0.1,
    highlight_mask=None,
    ticklabels=True,
    figsize=None,
    hists=True,
    facecolor=(0, 0.5, 0.5),
    edgecolor=(1, 1, 1),
    f_opacity=None,
    e_opacity=None,
    highlight_color="red",
):
    """Scatter-plot matrix with interactive capabilities.

    Parameters
    ----------
    data : (K,T) ndarray
        K variables, T timesteps
    variable_names : sequence of strings, optional
        Used to label the subplots.
    f_kwds : dictionary, optional
        Keyword arguments that are passed to plt.subplots.
    h_kwds : dictionary, optional
        Keyword arguments that are passed to the histogramm calls.
    s_kwds : dictionary, optional
        Keyword arguments that are passed to the scatter calls.
    opacity : float, optional
        Opacity used for edgecolor parameter of scatter call.
    highlight_mask : (T,) boolean ndarray, optional
        Mask of timesteps that will be highlighted in the scatter plots.
    ticklabels : boolean, optional
        Set to False to supress displaying x- and yticklabels.
    figsize : None or tuple of width, height, optional
        Size of the figure.
    hists : boolean, optional
        Plot histograms on the main diagonal.
    """
    cc = mpl.colors.ColorConverter()

    def switch_fc(artist, ind):
        fc = artist._facecolors
        if len(fc) == 1:
            fc = np.array(len(artist._offsets) * fc.tolist())
        fc[ind] = 1 - fc[ind]
        return fc

    def brush(event):
        """Highlight points in all plots."""
        # cache the collections. we will hopefully not get any more subplots
        if not hasattr(brush, "collections"):
            brush.collections = [
                sub.collections[0]
                for sub in fig.get_children()
                if hasattr(sub, "collections") and len(sub.collections) == 1
            ]
        # handle the click on a histogram bin
        if type(event.artist) is mpl.patches.Rectangle:
            event.artist._facecolor = tuple(
                1 - color_comp for color_comp in event.artist._facecolor
            )
            # find the indices of the points within the bin clicked on
            lower = event.artist._x
            upper = lower + event.artist._width
            values = data[event.artist.axes.var_i].ravel()
            ind = np.where((values > lower) & (values <= upper))
        else:
            # handle the click on a single scatter point
            ind = event.ind
        for col in brush.collections:
            col._facecolors = switch_fc(col, ind)
        fig.canvas.draw()

    f_kwds = {} if f_kwds is None else f_kwds
    h_kwds = {} if h_kwds is None else h_kwds
    s_kwds = {} if s_kwds is None else s_kwds
    f_opacity = opacity if f_opacity is None else f_opacity
    e_opacity = opacity if e_opacity is None else e_opacity
    data = np.asarray(data)
    n_variables = data.shape[0]
    fig, axes = square_subplots(n_variables, figsize=figsize, **f_kwds)

    for ii in range(n_variables):
        for jj in range(n_variables):
            if ii == jj:
                if hists:
                    axes[ii, jj].hist(
                        np.where(np.isnan(data[ii]), 0, data[ii]),
                        min(20, int(len(data[ii]) ** 0.5)),
                        picker=5,
                        density=True,
                        # want to achieve red when inverting
                        facecolor=cc.to_rgba(facecolor, alpha=0),
                        **h_kwds
                    )
                    # store the ii-index as an attribute to identify the
                    # variable later in the brush function
                    axes[ii, jj].var_i = ii
                else:
                    fig.delaxes(axes[ii, jj])
            else:
                if highlight_mask is None:
                    facecolors = cc.to_rgba(facecolor, alpha=f_opacity)
                else:
                    facecolors = np.empty((data.shape[1], 4))
                    facecolors[highlight_mask] = cc.to_rgba(
                        highlight_color, f_opacity
                    )
                    # want to achieve red when inverting
                    facecolors[~highlight_mask] = cc.to_rgba(
                        facecolor, f_opacity
                    )
                axes[ii, jj].scatter(
                    data[jj],
                    data[ii],
                    picker=5,
                    facecolors=facecolors,
                    edgecolors=cc.to_rgba(edgecolor, e_opacity),
                    **s_kwds
                )
            # show ticklabels only on the margins
            #                if (jj != 0) or (ii == jj):
            #                    axes[ii, jj].set_yticklabels("")
            #                if ii != n_variables - 1:
            #                    axes[ii, jj].set_xticklabels("")
            if variable_names and jj == 0:
                axes[ii, jj].set_ylabel(variable_names[ii])
            if variable_names and ii == n_variables - 1:
                axes[ii, jj].set_xlabel(variable_names[jj])
            if not ticklabels:
                axes[ii, jj].set_yticks([])
                axes[ii, jj].set_xticks([])

    fig.canvas.mpl_connect("pick_event", brush)
    return fig


def kde_gauss(
    dataset,
    evaluation_points=None,
    kernel_width=None,
    maxopt=500,
    return_width=False,
    verbose=False,
):
    """

    Parameter
    ---------
    dataset : (T,) ndarray
        input data as array of length T
    evaluation points : (N,) ndarray, optional if return_width=True
        N points as array (e.g. xx=np.linspace(-3,3,100)
    kernel_width : float, optional
        kernel_width (eg 0.3) if set, no MLM-routine to infer optimal kernel
        width
    maxopt : int, optional
        size of sample to optimize kernel_width, affects runtime strongly.
        depends on memorysize. max 1000 with 2GB mem
    return_width : boolean, optional
        Return the optimized kernel width and nothing else.
    verbose : boolean, optional
        Print information (also from the optimizer).

    Example
    -------
        dataset = np.array(np.random.normal(size=1e3))
        xx = np.linspace(-3,3, 1e3)
        plt.plot(xx,kde_gauss(dataset,xx))
    """

    def residual_matrix(x_vec, y_vec):
        """Returns a matrix with residuals (x:horizontal,y:vertical)"""
        x_vec, y_vec = list(map(np.asarray, (x_vec, y_vec)))
        return x_vec[None, :] - y_vec[:, None]

    def neglog_likelihood(kernel_width, dataset, verbose=False):
        """optimizing kernel width with MLM leave one out"""

        optMatrix = residual_matrix(dataset, dataset)
        if NE:
            pi = np.pi
            ne_str = (
                "1.0 / (sqrt(2 * pi) * kernel_width) / "
                "exp(optMatrix ** 2 / (2 * kernel_width ** 2))"
            )
            optMatrix = ne.evaluate(ne_str)
        else:
            preTerm = old_div(1.0, (np.sqrt(2 * np.pi) * kernel_width))
            optMatrix = old_div(
                preTerm,
                np.exp(old_div(optMatrix**2, (2 * kernel_width**2))),
            )
        nDataset = np.shape(dataset)[0]
        # sets diagonal to 0, i.e. leave-one-out method
        optMatrix.ravel()[:: nDataset + 1] = 0
        densities = old_div(np.sum(optMatrix, axis=1), float(nDataset - 1))
        # LN if <>0 for MLM
        d_sum = 0
        err = 0
        for d in densities:
            if d > 0:
                d_sum -= np.log(d)
            else:
                if err == 0:
                    if verbose:
                        print("LN(0) case do attend")
                    err = 1
                d_sum += 100  # not nice
        return d_sum

    dataset = np.asarray(dataset)
    # problem bei optimierung: d fluktuiert und haengt von maxopt ab
    # je hoeher maxopt desto kleiner d!
    dataset = np.sort(dataset)

    # optimizing kernel width if d=None
    if kernel_width is None:
        data_width = dataset.max() - dataset.min()
        d_0 = old_div(data_width, 10)
        if d_0 < 0.0001:
            d_0 = 0.0001
        if len(dataset) > 1000:
            fluct = True  # while values fluctuate, repeat iteration
            d_n = []  # list of d's
            d_act, d_old = d_0, 0
            n_min = 8 + np.sqrt(
                old_div(len(dataset), maxopt)
            )  # min nr of iterations
            n_act = 0

            while fluct or n_act <= n_min:
                dataset_sample = random.sample(tuple(dataset), maxopt)
                d_n.append(
                    optimize.fmin(
                        neglog_likelihood,
                        d_act,
                        args=(dataset_sample, verbose),
                        disp=verbose,
                    )[0]
                )
                d_act = old_div(sum(d_n), float(len(d_n)))
                # stop if fluct < 1%
                if old_div(abs(d_act - d_old), float(d_act)) < 0.01:
                    fluct = False
                if verbose:
                    print(d_act, d_old, d_n[-1])
                d_old = d_act
                n_act += 1
            kernel_width = d_act

        else:
            dataset_sample = dataset
            kernel_width = optimize.fmin(
                neglog_likelihood,
                d_0,
                args=(dataset_sample, verbose),
                disp=verbose,
            )
        if verbose:
            print("Kernelwidth = %f" % kernel_width)

    if return_width:
        return kernel_width

    evaluation_points = np.asarray(evaluation_points)

    if len(dataset) < len(evaluation_points):
        print(
            "Caution: you get more ev. points than input data\
        be aware of pseudo exactness"
        )

    # save kernel width, so it can be retrieved if anyone is interested
    kde_gauss.kernel_width = kernel_width

    # creating Matrix with residuals
    #    kdeMatrix = residual_matrix(dataset,evaluation_points)
    #    sparse_kde_mask = kdeMatrix < .001
    #    from scipy.sparse import lil_matrix
    #    sparse_kde = lil_matrix(kdeMatrix.shape)
    #    sparse_kde[sparse_kde_mask] = kdeMatrix[sparse_kde_mask]
    #     using Gaussian kernel
    #    import numexpr as ne
    #    preTerm = ne.evaluate("1.0 / ((2 * math.pi)**.5 * kernel_width)")
    #    kdeMatrix = preTerm / np.exp(kdeMatrix ** 2 / (2 * kernel_width ** 2))

    if len(dataset) * len(evaluation_points) > 1e7:
        parts = int(len(dataset) * len(evaluation_points) / 1e7) + 1
        brIncr = int(old_div(len(evaluation_points), parts))
        densities = np.array([])
        for i in range(parts + 1):
            if verbose:
                print("part %i of %i" % (i, parts))
            kdeMatrix = residual_matrix(
                dataset, evaluation_points[i * brIncr : (i + 1) * brIncr]
            )
            preTerm = old_div(1.0, (np.sqrt(2 * np.pi) * kernel_width))
            kdeMatrix = old_div(
                preTerm,
                np.exp(old_div(kdeMatrix**2, (2 * kernel_width**2))),
            )
            tmp_densities = old_div(
                np.sum(kdeMatrix, axis=1), float(len(dataset))
            )
            densities = np.hstack((densities, tmp_densities))
    else:
        kdeMatrix = residual_matrix(dataset, evaluation_points)
        preTerm = 1.0 / (np.sqrt(2 * np.pi) * kernel_width)
        with np.errstate(all="ignore"):
            kdeMatrix = preTerm / np.exp(
                kdeMatrix**2 / (2 * kernel_width**2)
            )
        # suming lines
        densities = np.sum(kdeMatrix, axis=1) / float(len(dataset))
    return densities


def hist(
    values,
    n_bins,
    dist=None,
    pdf=None,
    kde=False,
    fig=None,
    ax=None,
    discrete=False,
    figsize=None,
    legend=True,
    *args,
    **kwds
):
    """Plots a histogram and therotical or empirical densities."""
    try:
        if np.any(~np.isfinite(values)):
            warnings.warn("Non-finite values in values.")
    except TypeError:
        pass
    figsize = plt.rcParams["figure.figsize"] if figsize is None else figsize
    if ax is None:
        fig = plt.figure(figsize=figsize) if fig is None else fig
        axes = ax1 = fig.add_subplot(111)
    else:
        axes = ax1 = ax

    # the histogram of the data
    if discrete:
        values_2d = np.atleast_2d(values)
        bin_offset = -0.5 * values_2d.shape[0]
        for i, values in enumerate(values_2d):
            values = np.array(values)
            bins = np.arange(values.min(), values.max() + 1, dtype=int)
            bins = bins + bin_offset + i
            freqs = np.bincount(values.astype(int))
            freqs = freqs[freqs >= bins.min()]
            freqs = old_div(freqs.astype(float), values.size)
            ax1.vlines(bins, 0, freqs, linewidth=3)
            ax1.set_xlim(bins[0] - 1, bins[-1] + 1)
    else:
        bins = ax1.hist(
            values,
            n_bins,
            density=True,
            facecolor="grey",
            alpha=0.75,
            *args,
            **kwds
        )[1]

    ax1.set_ylabel("relative frequency")

    if not (isinstance(values, list) or values.ndim == 2):
        values_2d = (values,)
    else:
        values_2d = values

    if discrete:
        eva_points = bins
    else:
        eva_points = np.linspace(bins[0], bins[-1], 4 * n_bins)
    if kde:
        for val_i, values in enumerate(values_2d):
            density = kde_gauss(values, eva_points)
            ax1.plot(eva_points, density, label=("kde%d" % val_i))
    if dist:
        try:
            dist[0]
            dists = dist
        except TypeError:
            dists = (dist,)

        # the quantile part
        ax2 = ax1.twinx()
        axes = [ax1, ax2]
        for values in values_2d:
            # empirical cdf
            values_sort = np.sort(values)
            ranks_emp = old_div((0.5 + np.arange(len(values))), len(values))
            ax2.plot(values_sort, ranks_emp)
            pdf = []
            for dist in dists:
                if hasattr(dist, "fit"):
                    fitted_dist = dist(*dist.fit(values))
                else:
                    fitted_dist = dist
                pdf += [fitted_dist.pdf]
                # theoretical cdf
                ranks_theory = fitted_dist.cdf(eva_points)
                p_val = stats.kstest(values, fitted_dist.cdf, mode="asymp")[1]
                ax2.plot(
                    eva_points,
                    ranks_theory,
                    "--",
                    label=("%s p-value: %.1f%%" % (dist.name, p_val * 100)),
                )
                ax2.set_ylabel(r"cumulative frequency")
                ax2.set_ylim(0, 1)
                ax2.grid()

        if len(dists) == 1:
            if hasattr(dist, "parameter_names"):
                plt.title(
                    " ".join(
                        "%s:%.3f" % (par_name, par)
                        for par_name, par in zip(
                            dist.parameter_names, fitted_dist.params
                        )
                    )
                )
            elif hasattr(fitted_dist, "args"):
                plt.title(" ".join("%.3f" % par for par in fitted_dist.args))
        elif len(dists) > 1 and legend:
            plt.legend(loc="best")
    if pdf:
        try:
            pdf[0]
            pdfs = pdf
        except TypeError:
            pdfs = (pdf,)
        for pdf in pdfs:
            density_th = pdf(eva_points)
            if discrete:
                density_th *= len(values)
            ax1.plot(
                eva_points,
                density_th,
                "--o" if discrete else "--",
                linewidth=1,
                label="pdf",
            )

    if fig is not None:
        return fig, axes
    else:
        axes


def scale_yticks(event):
    """Automagically make room for yticklabels.
    Use it like this:
        fig = gcf()
        fig.canvas.mpl_connect('draw_event', scale_yticks)

    Stolen from the matplotlib-howto.  Slightly changed, so it is possible to
    separate the function from the calling code.
    http://matplotlib.sourceforge.net/faq/howto_faq.html\
    #automatically-make-room-for-tick-labels
    """
    labels = plt.gca().get_yticklabels()
    fig = plt.gcf()
    bboxes = []
    for label in labels:
        bbox = label.get_window_extent()
        # the figure transform goes from relative coords->pixels and we
        # want the inverse of that
        # bboxi = bbox.inverse_transformed(fig.transFigure)
        bboxi = bbox.transformed(fig.transFigure.inverted())
        bboxes.append(bboxi)

    # this is the bbox that bounds all the bboxes, again in relative
    # figure coords
    bbox = mpl.transforms.Bbox.union(bboxes)
    if fig.subplotpars.left < bbox.width:
        # we need to move it over
        fig.subplots_adjust(left=1.1 * bbox.width)  # pad a little
        fig.canvas.draw()
    return False


def yscale_subplots(fig=None, per_type=False, regrid=False):
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
        try:
            subplot.set_ylim(ymin=ymin[sub_type], ymax=ymax[sub_type])
        except TypeError:
            pass
        if regrid:
            subplot.set_rgrids(np.linspace(1e-6, ymax[sub_type], 10))


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    fig, ax = plt.subplots()
    (line,) = ax.plot(np.arange(10))
    ax.legend(
        [line, None],
        ("line", "find me"),
        handler_map={None, LegendSubtitleHandler()},
    )
    plt.show()
