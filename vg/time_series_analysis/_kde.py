"""
Not meant as a comprehensive Kernel Density Estimation package, but
simply some common ground for other modules that need a little bit of
KDE (distributions, seasonal_kde).
"""

import warnings

import numpy as np
from past.utils import old_div
from scipy import optimize, sparse

from vg import helpers as my, times


def distance_array(x_vec, y_vec, mask=None):
    """creating a matrix with residuals (x:horizontal,y:vertical)"""
    x_vec, y_vec = np.atleast_1d(x_vec, y_vec)
    x_slice = tuple(y_vec.ndim * [None] + [Ellipsis])
    y_slice = tuple([Ellipsis] + x_vec.ndim * [None])
    if mask is None:
        # return x_vec[None, :] - y_vec[:, None]
        return x_vec[x_slice] - y_vec[y_slice]
    else:
        return np.where(mask,
                        x_vec[None, :] - y_vec[:, None],
                        np.nan)


def distance_array_sparse(x_vec, y_vec, mask):
    """Creates a sparse residual matrix."""
    rows, cols = np.where(mask)
    dist_sparse = sparse.coo_matrix(((x_vec[rows] - y_vec[cols]),
                                     (rows, cols)),
                                    shape=(len(x_vec), len(y_vec)))
    return dist_sparse.T.tocsr()


def apply_kernel_ne(kernel_width, data, eval_points=None):
    if eval_points is None:
        _distances = distance_array(data, data)
    else:
        _distances = distance_array(data, eval_points)
    _pi = np.pi
    if _distances.ndim > 2:
        k_slice = ([Ellipsis] +
                   (_distances.ndim - kernel_width.ndim) * [None])
        kernel_width = kernel_width[tuple(k_slice)]
    densities = ne.evaluate(
        "1.0 / (sqrt(2 * _pi) * kernel_width) * " +
        "exp(-_distances ** 2 / (2 * kernel_width ** 2))")
    return densities


def apply_kernel_np(kernel_width, data, eval_points=None, mask=None,
                    recalc=False):
    if recalc or not hasattr(apply_kernel_np, "distances"):
        if eval_points is None:
            distances = distance_array(data, data)
            apply_kernel_np.distances = distances
        else:
            distances = distance_array(data, eval_points)
    else:
        distances = apply_kernel_np.distances
    if distances.ndim > 2:
        k_slice = ([Ellipsis] +
                   (distances.ndim - kernel_width.ndim) * [None])
        kernel_width = kernel_width[k_slice]
    return (1.0 /
            (np.sqrt(2 * np.pi) * kernel_width) *
            np.exp(-distances ** 2 / (2 * kernel_width ** 2)))


try:
    import numexpr as ne
    apply_kernel = apply_kernel_ne
    NE = True
except ImportError:
    warnings.warn("Could not import numexpr. Using numpy for KDE instead.")
    apply_kernel = apply_kernel_np
    NE = False


def kernel_density(kernel_width, data, eval_points=None, *args, **kwds):
    kernel_table = apply_kernel(kernel_width, data,
                                eval_points=eval_points, *args, **kwds)
    density = np.sum(kernel_table, axis=1) / len(data)
    if density.ndim > 1:
        density = np.sum(density, axis=1) / len(data)
    return density


def sum_log_density(kernel_width, data):
    density = apply_kernel(kernel_width, data)
    density.ravel()[::len(data) + 1] = 0
    density = np.sum(density, axis=1) / float(len(density) - 1)
    nonzero_mask = np.nonzero(density)
    if np.any(nonzero_mask):
        return -np.nansum(np.log(density[nonzero_mask]))
    else:
        return -np.inf


def scotts_rule(n_data, n_dim=3):
    """Scott's rule of thumb for kernel bandwidth."""
    return n_data ** (-1. / (n_dim + 4))


def silvermans_rule(x):
    """Silverman's rule of thumb for kernel bandwidth."""
    return (1.06 * np.std(x) * len(x) ** (-1. / 5))


def optimal_kernel_width(x):
    x0 = silvermans_rule(x)
    res = optimize.minimize(sum_log_density, x0, args=(x,),
                            bounds=([1e-2, None],),
                            method="L-BFGS-B",
                            # method="TNC",
                            # method="SLSQP",
                            )
    if res.success and not np.isclose(res.x, 1e-6):
        return np.squeeze(res.x)
    warnings.warn("Recursing in optimal_kernel_width")
    try:
        # try with a little added noise again
        return optimal_kernel_width(x + x0 / 1000 * np.random.randn(len(x)))
    except RecursionError:
        return x0


def apply_2d_kernel_(kernel_width, data, doys, circ, doy_width,
                     eval_points=None, eval_doys=None):
    if eval_points is None:
        eval_points = data
    if eval_doys is None:
        eval_doys = doys
    if NE:
        def kernel(dist, width):
            _pi = np.pi
            return ne.evaluate("(sqrt(2 * _pi) * width) ** -1 *"
                               "exp(-dist ** 2 / (2 * width ** 2))")
    else:
        kernel = lambda dist, width: \
            ((np.sqrt(2 * np.pi) * width) ** -1 *
             np.exp(old_div(-dist ** 2, (2 * width ** 2))))
    distances = distance_array(data, eval_points)
    densities = kernel(distances, abs(kernel_width))
    temp_distances = doyhour_distance(doys, eval_doys, circ=circ)
    time_dens = kernel(temp_distances, abs(doy_width))
    if NE:
        densities = ne.evaluate("densities * time_dens")
    else:
        densities *= time_dens
    return densities  # / (kernel_width * circ)


def apply_2d_kernel(kernels_width, data, hour_width, eval_points=None,
                    eval_doys=None):
    # data_width, hour_width = kernels_width
    data_width = kernels_width
    if eval_points is None:
        eval_points = data
    if NE:
        def kernel(dist, width):
            _pi = np.pi
            return ne.evaluate("(sqrt(2 * _pi) * width) ** -1 *"
                               "exp(-dist ** 2 / (2 * width ** 2))")
    else:
        kernel = lambda dist, width: \
            ((np.sqrt(2 * np.pi) * width) ** -1 *
             np.exp(old_div(-dist ** 2, (2 * width ** 2))))
    # distance in data dimension
    distances = distance_array(data.ravel(), eval_points.ravel())
    densities = kernel(distances, abs(data_width))
    # distance in the two temporal dimensions
    hour_dist, doy_dist = np.meshgrid(list(range(data.shape[1])),
                                      list(range(data.shape[0])))
    hour_middle = data.shape[0] // 2
    doy_middle = data.shape[1] // 2
    time_distances = np.empty_like(data)
    temp = ne.evaluate("sqrt((hour_dist - hour_middle) ** 2 + "
                       "(doy_dist - doy_middle) ** 2)")
    time_distances[:] = temp[..., None]
    time_dens = kernel(time_distances.ravel(), abs(hour_width))
    if NE:
        densities = ne.evaluate("densities * time_dens")
    else:
        densities *= time_dens
    return densities


def gaussian_kernel(x, width):
    return (1. / (np.sqrt(2 * np.pi) * width) *
            np.exp(old_div(-x ** 2, (2 * width ** 2))))


def gaussian_kernel_ne(x, width):
    _pi = np.pi
    return ne.evaluate("(1. / (sqrt(2 * _pi) * width) *" +
                       "exp(-x ** 2 / (2 * width ** 2)))")


@my.cache("dists", "rows", "cols")
def apply_sparse_kernel(kernel, widths, data, mask, eval_points=None,
                        **k_kwds):
    """Apply a given kernel on parts of data indicated by a mask.

    Parameter
    ---------
    kernel : callable
    widths : float or 1d array
    data : 1d array
    mask : 2d boolean array
        Indicates which pairs in data are close enough to each other in order
        to influence each others density.
    eval_points : None or 1d array, optional
        Where to evaluate kernel densities.
    **k_kwds : optional
        Extra key-word arguments are passed into the supplied kernen callable.

    Returns
    -------
    dens : sparse.csr matrix
    """
    self = apply_sparse_kernel
    x_vec = data
    if eval_points is None:
        y_vec = data
    else:
        y_vec = eval_points
    # this is a poor man's cache invalidation!
    if self.dists is None or self.dists.size != data.size ** 2:
        # warning: this is a bit hacky! cache invalidation is a bitch
        if self.rows is None or np.sum(mask) != len(self.rows):
            self.cols, self.rows = np.where(mask)
        self.dists = x_vec[self.rows] - y_vec[self.cols]
    try:
        widths = widths[self.rows]
    except TypeError:
        pass
    dens = sparse.coo_matrix((kernel(self.dists, widths, **k_kwds),
                              (self.rows, self.cols)),
                             shape=(len(x_vec), len(y_vec)))
    return dens.T.tocsr()


def doy_distance_int(doys1, doys2, doy_width=1):
    """Distance between two times in terms of integer doys.

    doys1 : decimal doy or ndarray of decimal doys
    doys2 : decimal doy or ndarray of decimal doys
    doy_width : float
        scaling factor for distance in the doy dimension
    """
    doys1, doys2 = list(map(np.atleast_1d, (doys1, doys2)))
    doy_distance = np.squeeze(times.doy_distance(doys1[None, :],
                                                 doys2[:, None]))
    return doy_width * doy_distance


def hour_distance_using_doys(doys1, doys2, circ=1):
    """Distance between two times in terms of hours.

    doys1 : decimal doy or ndarray of decimal doys
    doys2 : decimal doy or ndarray of decimal doys
    circ : float
        minor circumference of torus. scales the hour distance in relation to
        the doy distance.
    """
    doys1, doys2 = list(map(np.atleast_1d, (doys1, doys2)))
    hours1, hours2 = [(doy % 1) * 24 for doy in (doys1, doys2)]
    hour_distance = np.squeeze(times.hour_distance(hours1[None, :],
                                                   hours2[:, None]))
    return circ * hour_distance


def doyhour_distance(doys1, doys2, doy_width=1, circ=1):
    """Distance between two times treating the hour as a second dimension.
    Similar but not exactly like the distance on the surface of a torus.

    Parameter
    ---------
    doys1 : decimal doy or ndarray of decimal doys
    doys2 : decimal doy or ndarray of decimal doys
    doy_width : float
        scaling factor for distance in the doy dimension
    circ : float
        minor circumference of torus. scales the hour distance in relation to
        the doy distance.
    """
    # doys1, doys2 = map(np.atleast_1d, (doys1, doys2))
    # hours1, hours2 = map(lambda doy: (doy % 1) * 24, (doys1, doys2))
    # doy_distance = np.squeeze(times.doy_distance(doys1[None, :],
    #                                              doys2[:, None]))
    # hour_distance = np.squeeze(times.hour_distance(hours1[None, :],
    #                                                hours2[:, None]))
    # return np.sqrt((doy_width * doy_distance) ** 2 +
    #                (circ * hour_distance / 24.) ** 2)
    return np.sqrt(doy_distance_int(doys1, doys2, doy_width=doy_width) ** 2 +
                   (hour_distance_using_doys(doys1, doys2, circ=circ)
                    / 24.) ** 2)


def doyhour_distance_dt(dtimes1, dtimes2, circ):
    """Distance between two times treating the hour as a second dimension.
    Similar but not exactly like the distance on the surface of a torus.

    Parameter
    ---------
    dtimes1 : datetime object or ndarray of datetime objects
    dtimes2 : datetime object or ndarray of datetime objects
    circ : float
        minor circumference of torus. scales the hour distance in relation to
        the doy distance.
    """
    doys1, doys2 = list(map(times.datetime2doy, (dtimes1, dtimes2)))
    # TODO: for speed: calculate the hours from the doys without calling
    # datetime2hour
    hours1, hours2 = list(map(times.datetime2hour, (dtimes1, dtimes2)))
    doy_distance = times.doy_distance(doys1, doys2)
    hour_distance = times.hour_distance(hours1, hours2)
    return np.sqrt(doy_distance ** 2 + (circ * hour_distance / 24.) ** 2)
