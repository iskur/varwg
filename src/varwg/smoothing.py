"""
1dim smoothing and moving moments (:mod:`smoothing`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: varwg.smoothing

.. autosummary::
   :nosignatures:
   :toctree: generated/

   smooth
   variance
   std
   skew
   median
   min
   max
   percentile
   autocorr
   corr
   crosscorr

"""

import numpy as np
from functools import wraps

_pars_descr = """

    Parameters
    ----------
    data : 1d ndarray
    window_len : int, optional
        Length of the moving window.
    periodic : boolean, optional
        Assumes the data is given as one period and appends window_len elements
        from the beginning at the end and window_len elements from the end to
        the beginning. When no trend exists, this nicely handles the estimation
        at the boundaries.
    l_value : float, optional
        Will be appended at the beginning. Might help to better estimate the
        first window_len elements.
    r_value : float, optional
        Will be appended at the end. Might help to better estimate the last
        window_len elements.
    loo : boolean, optional
        leave one out
        Estimates the percentile for data[t] without data[t].
    no_future: boolean, optional
        only use past values for smoothing"""
_pars_descr_ddof = """
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        Default: {ddof}
        (passed on via **kwds)"""


def _docstring_pars(ddof=None):
    """The functions calling _convolve all have the same signature. So we can
    save a lot of docstring duplication if we update all their docstrings with
    the same parameter description."""

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)

        wrapper.__doc__ = func.__doc__ + _pars_descr
        if type(ddof) == int:
            wrapper.__doc__ = wrapper.__doc__ + _pars_descr_ddof.format(
                ddof=ddof
            )
        return wrapper

    # this makes the decorator also work when used as
    # @_docstring_pars
    # instead of
    # @_docstring_pars(1)
    # or
    # @_docstring_pars()
    return decorate(ddof) if callable(ddof) else decorate


@_docstring_pars(ddof=0)
def _convolve(
    data,
    window_len=10,
    window_function=np.hanning,
    periodic=False,
    l_value=None,
    r_value=None,
    ddof=0,
    loo=False,
    no_future=False,
    **kwds
):
    """Common ground for the 1dim-smoothing functions."""
    window_len = int(round(window_len))
    if no_future:
        # assure uneven length of window and zero out the future later
        window_len += 1 - window_len % 2

    if window_function == "flat":  # moving average
        smooth_vector = np.ones(window_len, "d")
    else:
        smooth_vector = window_function(window_len)

    if loo:
        # leave one out
        smooth_vector[int(window_len / 2)] = 0

    if no_future:
        # no future no future for you
        # god save the queen
        smooth_vector[int(np.ceil(window_len / 2.0))] = 0

    if periodic:
        ldata = np.array(data[-window_len:])
        rdata = np.array(data[:window_len])
    else:
        if l_value is None:
            l_value = np.average(data[:window_len])
        if r_value is None:
            r_value = np.average(data[-window_len - 1 :])
        ldata = np.array([l_value] * window_len)
        rdata = np.array([r_value] * window_len)
    data_window = np.concatenate((ldata, data, rdata))

    smoothed = np.convolve(
        smooth_vector / (smooth_vector.sum() - ddof), data_window, mode="same"
    )

    return smoothed[window_len:-window_len]


@_docstring_pars(ddof=0)
def smooth(
    data,
    window_len=10,
    window_function=np.hanning,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
    **kwds
):
    """1-dimensional smoothing
    Simplified & modified http://www.scipy.org/Cookbook/SignalSmooth
    """
    return _convolve(**locals())


@_docstring_pars(ddof=1)
def variance(
    data,
    window_len=10,
    window_function=np.hanning,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
    **kwds
):
    """Moving variance."""
    means = smooth(**locals())
    data = (np.copy(data) - means) ** 2
    return _convolve(ddof=1, **locals())


@_docstring_pars(ddof=1)
def std(
    data,
    window_len=10,
    window_function=np.hanning,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
    **kwds
):
    """Moving standard deviation."""
    means = smooth(**locals())
    data = (data.copy() - means) ** 2
    return np.sqrt(_convolve(ddof=1, **locals()))


@_docstring_pars(ddof=2)
def skew(
    data,
    window_len=10,
    window_function=np.hanning,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
):
    """Moving skew."""
    means = smooth(**locals())
    variances = variance(**locals())
    data = ((data.copy() - means) / variances**0.5) ** 3
    return _convolve(ddof=2, **locals())


@_docstring_pars
def _roll_data(
    data,
    window_len=10,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
    **kwds
):
    """This prepares data for functions like min, max, percentile.
    Use smooth for the mean."""
    window_len = int(window_len)
    # this is how i roll!
    if periodic:
        ldata = np.array(data[-window_len:])
        rdata = np.array(data[:window_len])
    else:
        if l_value is None:
            l_value = np.average(data[:window_len])
        if r_value is None:
            r_value = np.average(data[-window_len - 1 :])
        ldata = np.array([l_value] * window_len)
        rdata = np.array([r_value] * window_len)
    data_window = np.concatenate((ldata, data, rdata))

    if no_future:
        shifts = list(range(window_len + 1))
    else:
        shifts = list(range(-window_len, window_len + 1))
    if loo:
        shifts.remove(0)

    data_rolled = np.array([np.roll(data_window, shift) for shift in shifts])
    data_rolled = data_rolled.reshape(len(shifts), len(data_window))
    return data_rolled


@_docstring_pars
def min(
    data,
    window_len=10,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
    **kwds
):
    """Moving minimum."""
    return np.min(_roll_data(**locals()), axis=0)[window_len:-window_len]


@_docstring_pars
def max(
    data,
    window_len=10,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
    **kwds
):
    """Moving maximum."""
    return np.max(_roll_data(**locals()), axis=0)[window_len:-window_len]


@_docstring_pars
def maxdiff(
    data,
    window_len=10,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
    **kwds
):
    """Moving maximum difference."""
    diffed = np.diff(_roll_data(**locals()), axis=0)
    return np.max(diffed, axis=0)[window_len:-window_len]


@_docstring_pars
def mindiff(
    data,
    window_len=10,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
    **kwds
):
    """Moving minimum difference."""
    diffed = np.diff(_roll_data(**locals()), axis=0)
    return np.min(diffed, axis=0)[window_len:-window_len]


@_docstring_pars
def percentile(
    data,
    perc,
    window_len=10,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
):
    """Moving percentile."""
    data_rolled = _roll_data(**locals())
    return np.percentile(data_rolled, perc, axis=0)[window_len:-window_len]


@_docstring_pars
def median(
    data,
    window_len=10,
    periodic=False,
    l_value=None,
    r_value=None,
    loo=False,
    no_future=False,
    **kwds
):
    """Moving median."""
    return percentile(
        data,
        50,
        window_len,
        periodic,
        l_value,
        r_value,
        loo,
        no_future=no_future,
    )


def corr(data1, data2, window_len=100):
    """Moving correlation. Returned array has the same length as data.
    This is achieved by assuming a special kind of periodicity: end of data is
    prepended to the beginning. You might be better off ignoring the first
    window_len elements.

    Parameters
    ----------
    data1 : 1dim ndarray
    data2 : 1dim ndarray
        This array will be shifted back by 'lag' steps
    window_len : int, optional
        window length, need i say more?
    """
    return crosscorr(data1, data2, lag=0, window_len=window_len)


def crosscorr(data1, data2, lag=1, window_len=100):
    """Moving crosscorrelation. Returned array has the same length as data.
    This is achieved by assuming a special kind of periodicity: end of data is
    prepended to the beginning. You might be better off ignoring the first
    window_len elements.

    Parameters
    ----------
    data1 : 1dim ndarray
    data2 : 1dim ndarray
        This array will be shifted back by 'lag' steps
    lag : int, optional
        0 works and gives correlations instead of crosscorrelations.
    window_len : int, optional
        window length, need i say more?
    """
    # naive implementation. there is a faster, more memory-friendly way I
    # guess

    if lag == 0:
        slice1 = slice2 = slice(None)
    else:
        slice1 = (slice(None), slice(lag, None))
        slice2 = (slice(None), slice(None, -lag))

    # do not know how to do this using a convolution, so i resort to
    # broadcasting
    # the following gives a (data_length, window_length) array
    data1_rolled = np.array(
        [
            np.roll(data1, window_len - 1 - shift)[:window_len]
            for shift in range(len(data1))
        ]
    )
    data2_rolled = np.array(
        [
            np.roll(data2, window_len - 1 - shift)[:window_len]
            for shift in range(len(data2))
        ]
    )
    means1 = data1_rolled.mean(axis=1)[:, np.newaxis]
    means2 = data2_rolled.mean(axis=1)[:, np.newaxis]
    covs = (data1_rolled[slice1] - means1) * (data2_rolled[slice2] - means2)
    return (
        np.mean(covs, axis=1)
        / (
            np.std(data1_rolled, ddof=1, axis=1)
            * np.std(data2_rolled, ddof=1, axis=1)
        ),
    )


def autocorr(data, lag=1, window_len=100):
    """Moving autocorrelation. Returned array has the same length as data. This
    is achieved by assuming a special kind of periodicity: end of data is
    prepended to the beginning. You might be better off ignoring the first
    window_len elements.

    Parameters
    ----------
    data : 1dim ndarray
    lag :  int, optional
    window_len : int, optional
        window length, need i say more?
    """
    # we could call crosscorr(data, data) here, which would be boon for
    # abstraction, but let's be kind and save some redundant cpu cycles.

    # do not know how to do this using a convolution, so i resort to
    # broadcasting
    # the following gives a (data_length, window_length) array, i.e. an array
    # with all windows
    data_rolled = np.array(
        [
            np.roll(data, window_len - 1 - shift)[:window_len]
            for shift in range(len(data))
        ]
    )
    means = data_rolled.mean(axis=1)[:, np.newaxis]
    covs = (data_rolled[:, :-lag] - means) * (data_rolled[:, lag:] - means)
    return np.mean(covs, axis=1) / np.var(data_rolled, ddof=1, axis=1)
