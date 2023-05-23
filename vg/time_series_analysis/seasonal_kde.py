from __future__ import division, print_function

import warnings

import matplotlib.pyplot as plt
import numpy as np
from builtins import range, zip
from past.utils import old_div
from scipy import interpolate, optimize, stats
from scipy.integrate import cumtrapz
from tqdm import tqdm

from vg import helpers as my, times
from vg.smoothing import smooth
from vg.time_series_analysis import _kde as kde, seasonal


try:
    from multiprocessing import cpu_count
    import numexpr as ne
    ne.set_num_threads(min(64, cpu_count()))
    NE = True
except ImportError:
    warnings.warn("Could not import numexpr. Using numpy for KDE instead.")
    NE = False


def array_gen(scalar):
    return lambda tt: scalar * np.ones_like(np.atleast_1d(tt))


class SeasonalKDE(seasonal.Seasonal):

    def __init__(self, data, datetimes, solution=None, doy_width=15,
               nx=1e3, fixed_pars=None, verbose=True, freibord=0,
               smooth_len=None, kill_leap=False):
        """
        nx is the number of data points used to estimate the quantiles.
        lower- and upper bound must be functions that accept a doy. If None
        smoothed seasonal minima and maxima of data will be used.

        Parameters
        ----------
        data : 1dim ndarray
            may contain nans.
        datetimes : sequence of datetime objects
        solution : sequence
            contains kernel widths and quantile_grid as returned from
            SeasonalKDE.fit
        doy_width :  int
            kernel width in time direction.
        nx : int
            number of datapoints for each doy to base the pdf, cdf and ppf
            estimates on
        fixed_pars : dict
            with 'l' and or 'u' mapping to a function that accepts doys and
            returns lower and upper bounds.
        verbose : boolean
        """
        super().__init__(data, datetimes, kill_leap)
        self.doy_width = doy_width
        self.nx = int(nx)
        self.verbose = verbose
        self.freibord = freibord
        if smooth_len is None:
            self.smooth_len = 3 * self.doy_width
        else:
            self.smooth_len = smooth_len

        # those will be populated by cached properties
        self._doy_mask = self._doy_mask_dict = self._kernel_widths = None
        self._density_grid = self._x_grid = self._quantile_grid = None
        self._cdf_interp_per_day = self._ppf_interp_per_day = None
        self._data_bounds = self._lower_bound = self._upper_bound = None

        if solution is not None:
            self.solution = solution

        if fixed_pars is None:
            fixed_pars = {}

        try:
            fixed_pars["l"] = fixed_pars["lc"]
        except KeyError:
            pass
        if "l" in fixed_pars:
            self.lower_bound = fixed_pars["l"]
        try:
            fixed_pars["u"] = fixed_pars["uc"]
        except KeyError:
            pass
        if "u" in fixed_pars:
            self.upper_bound = fixed_pars["u"]

        upper = self.upper_bound(self.doys)
        upper_ii = self.data >= upper
        if np.any(upper_ii):
            self.data[upper_ii] = (np.sign(upper[upper_ii]) *
                                   np.abs(upper[upper_ii]) * (1 - 1e-3))

        lower = self.lower_bound(self.doys)
        lower_ii = self.data <= lower
        if np.any(lower_ii):
            self.data[lower_ii] = (np.sign(lower[lower_ii]) *
                                   np.abs(lower[lower_ii]) * (1 + 1e-3))

    @property
    def data_bounds(self):
        """
        Here for optimization: We do not let lower and upper bound make
        a long loop over the data for min and max resp. We do it here once and
        cache it.
        """
        if self._data_bounds is None:
            doy_mask = np.isclose(self.doys, self.doys_unique[:, np.newaxis])
            self._data_bounds = \
                np.array([(np.nanmin(self.data[doy_ii]),
                           np.nanmax(self.data[doy_ii]))
                          for doy_ii in doy_mask])
        return self._data_bounds

    def lower_bound(self, doy):
        if self._lower_bound is None:
            data_mins, data_maxs = self.data_bounds.T
            data_mins_smooth = \
                smooth(data_mins, self.smooth_len, periodic=True)
            # we have to lower the smoothed mins below the actual data mins
            diffs = data_mins_smooth - data_mins
            # ideal would be a kernel width, but we do not have those yet
            diff_thresh = (old_div((data_maxs.mean() - data_mins.mean()),
                           (old_div(len(self.data), len(self.doys_unique)))))
            while np.any(diffs > -diff_thresh):
                sub = np.zeros_like(diffs)
                argmax = np.argmax(diffs)
                diff_max = diffs[argmax]
                sub[argmax] = diff_max
                sub = smooth(sub, self.smooth_len, periodic=True)
                sub *= (1 + diff_thresh) * self.smooth_len / diff_max
                data_mins_smooth -= sub
                diffs = data_mins_smooth - data_mins
            self._lower_bound = data_mins_smooth
        doy = np.rint(old_div((doy - 1), self.timestep)).astype(int)
        return self._lower_bound[doy]

    def upper_bound(self, doy):
        if self._upper_bound is None:
            data_mins, data_maxs = self.data_bounds.T
            data_maxs_smooth = \
                smooth(data_maxs, self.smooth_len, periodic=True)
            # we have to lift the smoothed maxs above the actual data maxs
            diffs = data_maxs - data_maxs_smooth
            # ideal would be a kernel width, but we do not have those yet
            diff_thresh = ((data_maxs.mean() - data_mins.mean()) /
                           (len(self.data) / len(self.doys_unique)))
            while np.any(diffs > -diff_thresh):
                add = np.zeros_like(diffs)
                argmax = np.argmax(diffs)
                diff_max = diffs[argmax]
                if np.isclose(diff_max, 0):
                    diff_max += 1e-6
                add[argmax] = diff_max
                add = smooth(add, self.smooth_len, periodic=True)
                add *= (1 + diff_thresh) * self.smooth_len / diff_max
                data_maxs_smooth += add
                diffs = data_maxs - data_maxs_smooth
            self._upper_bound = data_maxs_smooth + self.freibord
        doy = np.rint(old_div((doy - 1), self.timestep)).astype(int)
        return self._upper_bound[doy]

    @property
    def doy_mask(self):
        """Returns a (n_unique_doys, len(data)) ndarray"""
        if self._doy_mask is None:
            doy_width, doys = self.doy_width, self.doys
            self._doy_mask = \
                np.empty((len(self.doys_unique), len(self.data)), dtype=bool)
            for doy_i, doy in enumerate(self.doys_unique):
                ii = (doys > doy - doy_width) & (doys <= doy + doy_width)
                if (doy - doy_width) < 0:
                    ii |= doys > (365. - doy_width + doy)
                if (doy + doy_width) > 365:
                    ii |= doys < (doy + doy_width - 365.)
                self._doy_mask[doy_i] = ii
        return self._doy_mask

    @property
    def doy_mask_dict(self):
        if self._doy_mask_dict is None:
            self._doy_mask_dict = {}
            for row_i, row in enumerate(self.doy_mask):
                self._doy_mask_dict[self.doys_unique[row_i]] = row
        return self._doy_mask_dict

    @property
    def x_grid(self):
        if self._x_grid is None:
            xx = np.empty((len(self.doys_unique), self.nx))
            for doy_i, doy in enumerate(self.doys_unique):
                xx[doy_i] = np.squeeze(
                    np.linspace(self.lower_bound(doy),
                                self.upper_bound(doy),
                                self.nx))
            self._x_grid = xx
        return self._x_grid

    @property
    def density_grid(self):
        if self._density_grid is None:
            densities = np.zeros_like(self.x_grid)
            for doy_i, doy in enumerate(self.doys_unique):
                kernel_width = self.kernel_widths[doy_i]
                ii = self.doy_mask_dict[doy]
                x = self.x_grid[doy_i]
                densities[doy_i] = \
                    self.density_per_doy(kernel_width, self.data[ii],
                                         self.doys[ii], doy, x)
            self._density_grid = densities
            # HACK the following will normalize the densities
            self.quantile_grid
        return self._density_grid

    @property
    def quantile_grid(self):
        if self._quantile_grid is None:
            quantiles = cumtrapz(y=self.density_grid, x=self.x_grid,
                                 axis=1, initial=0)
            # if there is only zero-valued data at some doys
            quantiles[:, -1] = np.where(quantiles[:, -1] != 0,
                                        quantiles[:, -1],
                                        1)
            # this normalizes the quantiles
            quantiles *= (quantiles[:, -1] ** -1)[:, np.newaxis]
            # just making it sure...
            quantiles[:, 0] = 0
            quantiles[:, -1] = 1.
            self._quantile_grid = quantiles
        return self._quantile_grid

    @quantile_grid.setter
    def quantile_grid(self, grid):
        self._quantile_grid = grid

    @property
    def cdf_interp_per_day(self):
        if self._cdf_interp_per_day is None:
            self._cdf_interp_per_day = \
                [interpolate.interp1d(self.x_grid[doy_i], quantiles)
                 for doy_i, quantiles in enumerate(self.quantile_grid)]
        return self._cdf_interp_per_day

    @property
    def ppf_interp_per_day(self):
        if self._ppf_interp_per_day is None:
            self._ppf_interp_per_day = \
                [interpolate.interp1d(quantiles, self.x_grid[doy_i],
                                      kind="nearest",
                                      assume_sorted=True)
                 if self.x_grid[doy_i, -1] > 0
                 else lambda q, *args, **kwds: 0
                 for doy_i, quantiles in enumerate(self.quantile_grid)]
        return self._ppf_interp_per_day

    def density_per_doy(self, kernel_width, data, doys, doy_middle,
                        eval_data=None, leave_one_out=False):
        densities = kde.apply_kernel(kernel_width, data, eval_data,
                                     recalc=True)
        doy_scale = (1. - times.doy_distance(doy_middle, doys) /
                     (self.doy_width + 1)) / (self.doy_width - 1)
        densities *= doy_scale
        # we want the densities at a specific doy to integrate to one
        densities *= len(doys) / self.doy_width
        if leave_one_out:
            # sets main diagonal to 0 (exploiting the fact that ravel returns
            # a view)
            densities.ravel()[::len(data) + 1] = 0
        return np.sum(densities, axis=1) / float(len(densities) - 1)

    def sum_log_density(self, kernel_width, data, doys, doy_middle):
        """objective function to optimize kernel width with MLM leave one
        out"""
        densities = self.density_per_doy(np.abs(kernel_width), data, doys,
                                         doy_middle, leave_one_out=True)
        return -np.sum(np.log(densities[densities > 0]))

    def fit(self, silverman=False):
        """This estimates the kernel widths per doy AND returns the
        interpolating functions for cdf and ppf."""
        kernel_widths = np.empty(len(self.doys_unique))
        # if the data is very coarse, add a little noise
        data = np.copy(self.data)
        for doy_i, doy in tqdm(enumerate(self.doys_unique),
                               total=kernel_widths.size,
                               disable=(not self.verbose)):
            ii = self.doy_mask[doy_i]
            data_ = data[ii]
            if silverman:
                kernel_widths[doy_i] = kde.silvermans_rule(data_)
                continue
            if doy_i == 0:
                x0 = kde.silvermans_rule(data_) / 2
            else:
                x0 = kernel_widths[doy_i - 1]
            result = \
                optimize.minimize(self.sum_log_density, x0,
                                  args=(data_, self.doys[ii], doy),
                                  bounds=[(1e-9, None)],
                                  # method="L-BFGS-B",
                                  # method="TNC",
                                  # method="SLSQP",
                                  # options=dict(disp=True),
                                  options=dict(disp=False),
                                  )
            kernel_width = result.x
            # we want to avoid very small kernel widths as that causes
            # problems later on
            if kernel_width < 1e-8:
                kernel_width = kde.silvermans_rule(data_)
            kernel_widths[doy_i] = kernel_width

        if self.verbose:
            print()
        kernel_widths = np.abs(kernel_widths)

        nan_ii = np.where(np.isnan(kernel_widths))[0]
        if len(nan_ii):
            left_ii = np.maximum(nan_ii - 2, 0)
            right_ii = np.minimum(nan_ii + 2, len(kernel_widths) - 1)
            kernel_widths[left_ii] = np.nan
            kernel_widths[right_ii] = np.nan
            kernel_widths = my.interp_nan(kernel_widths, pad_periodic=True)
            assert len(kernel_widths) == len(self.doys_unique)

        kernel_widths = smooth(kernel_widths, self.doy_width, periodic=True)
        self._kernel_widths = kernel_widths
        return kernel_widths, self.quantile_grid

    def chi2_test(self, k=5):
        """Chi-square goodness-of-fit test.
        H0: The given data **x** follows **distribution** with parameters
            aquired by ``func::SeasonalDist.fit``
        To side-step complications arising from having a different
        distribution for every doy, we test whether the quantiles (which are
        deseasonalized) are uniformly distributed.

        Parameters
        ----------
        k : int
            Number of classes (bins)

        Returns
        -------
        p_value : float
        """
        quantiles = self.cdf(self.solution)
        n = len(quantiles)
        observed = np.histogram(quantiles, k)[0]
        expected = old_div(float(n), k)
        chi_test = np.sum(old_div((observed - expected) ** 2, expected))
        # degrees of freedom:
        dof = k - 1
        return stats.chi2.sf(chi_test, dof)

    @property
    def kernel_widths(self):
        if self._kernel_widths is None:
            self.fit()
        return self._kernel_widths

    @kernel_widths.setter
    def kernel_widths(self, widths):
        self._kernel_widths = widths

    @property
    def solution(self):
        return self.kernel_widths, self.quantile_grid

    @solution.setter
    def solution(self, solution):
        self.kernel_widths, self.quantile_grid = solution

    def pdf(self, solution, data, doys):
        data, doys = np.atleast_1d(data, doys)
        densities = np.empty((len(doys), len(data)))
        for doy in doys:
            doy_i = my.val2ind(self.doys_unique, doy)
            kernel_width = self.kernel_widths[doy_i]
            ii = self.doy_mask_dict[doy]
            densities[doy_i] = self.density_per_doy(kernel_width,
                                                    self.data[ii],
                                                    self.doys[ii],
                                                    doy,
                                                    data)
        return densities

    def cdf(self, solution=None, x=None, doys=None):
        if solution is not None:
            self.solution = solution
        if x is None:
            x = self.data
        if doys is None:
            doys = self.doys

        x, doys = np.atleast_1d(x, doys)
        quantiles = np.full_like(x, np.nan)
        for ii, (doy, x_single) in enumerate(zip(doys, x)):
            doy_i = my.val2ind(self.doys_unique, doy)
            try:
                quantiles[ii] = self.cdf_interp_per_day[doy_i](x_single)
            except ValueError:
                if x_single < self.lower_bound(self.doys_unique[doy_i]):
                    quantiles[ii] = np.nan
                    continue
                elif x_single > self.upper_bound(self.doys_unique[doy_i]):
                    quantiles[ii] = np.nan
                    continue
        if np.any(np.isnan(quantiles)):
            quantiles = my.interp_nan(quantiles, max_interp=3)
        return quantiles

    def ppf(self, solution, quantiles, doys=None, mean_shift=None):
        if solution is not None:
            self.solution = solution
        if doys is None:
            doys = self.doys
        if mean_shift is not None:
            raise NotImplementedError(f"mean_shift not implemented"
                                      f"for {type(self)} yet.")
        quantiles, doys = np.atleast_1d(quantiles, doys)
        # for the purpose of distribution parameters: assume February
        # 29th behaves as February 28th
        doys = self.duplicate_feb28(doys)
        x = np.empty_like(quantiles)
        for ii, (doy, quantile_single) in enumerate(zip(doys, quantiles)):
            # doy_i = my.val2ind(self.doys_unique, doy)
            doy_i = int((doy - 1) / self.dt)
            x[ii] = self.ppf_interp_per_day[doy_i](quantile_single)
        return x

    def scatter_pdf(self, solution, n_sumup=24, figsize=None, title=None,
                    opacity=.25, plot_kernel_width=False, s_kwds=None):
        if s_kwds is None:
            s_kwds = dict(marker="o")
        doys = \
            self.doys_unique.repeat(self.nx).reshape(len(self.doys_unique), -1)
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.contourf(doys, self.x_grid / n_sumup, self.density_grid, 15)
        plt.set_cmap("coolwarm")
        ax1.scatter(self.doys, self.data / n_sumup, facecolors=(0, 0, 0, 0),
                    edgecolors=(0, 0, 0, opacity), **s_kwds)
        ax1.set_ylim(self.x_grid.min() / n_sumup, self.x_grid.max() / n_sumup)

        if plot_kernel_width:
            ax2 = ax1.twinx()
            ax2.plot(self.doys_unique, self.kernel_widths.T.ravel(), "r-",
                     label="Kernel width")
            ax2.set_ylim(self.kernel_widths.min(), self.kernel_widths.max())

        ax1.set_xlim(0, 366)
        self._set_monthly_ticks(ax1)

        ax1.grid()
        if title is not None:
            ax1.set_title(title)
        return fig, ax1

    def scatter_cdf(self, solution, n_sumup=1, figsize=None, title=None):
        doys = self.doys_unique.repeat(1e3).reshape(len(self.doys_unique), 1e3)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.gca()
        co = ax1.contourf(doys, self.x_grid, self.quantile_grid, 15)
        plt.colorbar(co)
        plt.scatter(self.doys, old_div(self.data, n_sumup), marker="o",
                    facecolors=(0, 0, 0, 0), edgecolors=(0, 0, 0, .5))
        ax1.set_ylim(self.x_grid.min(), self.x_grid.max())
        ax2 = plt.gca().twinx()
        ax2.plot(self.doys_unique, self.kernel_widths, label="Kernel width")
        ax2.set_ylim(self.kernel_widths.min(), self.kernel_widths.max())

        plt.xlim(0, len(doys))
        plt.xlabel("Day of Year")
        plt.grid()
        plt.legend()
        if title is not None:
            plt.title(title)
        return fig


def doy_hour_fft(data, dtimes, order=4):
    """Remove yearly and daily cycle.

    Parameter
    ---------
    data : 1d array
    dtimes : 1d array of datetime objects
    order : int, optional
        number of frequencies to use

    Returns
    -------
    xx : 1d array
    fft_pars : 24d array
        fast fourier parameters for each hour.
    """
    xx = np.copy(data)
    hours_int = np.array([dtime.hour for dtime in dtimes])
    fft_pars = []
    # approximate cycles of values for a given hour
    for hour in range(24):
        ii = hours_int == hour
        values = data[ii]
        fft_par = np.fft.rfft(values)
        pars_below = np.argsort(np.abs(fft_par))[:len(fft_par) - order - 1]
        par = np.copy(fft_par)
        par[pars_below] = 0
        fft_pars += [par]
        xx[ii] -= np.fft.irfft(par, len(values))
    return xx, np.array(fft_pars)


def fft2par(fft_pars, doys, ifft_func=None, period_length=None,
            lower_bound=None):
    """Converts from the frequency into the time domain.
    The output is repeated for as many periods underlying the doys (think of
    years).

    Parameter
    ---------
    fft_pars : ndarray
        As returned from numpy.fft.fft (or its siblings).
    doys : float 1d array
        Doys associated with the output.
    ifft_func : callable or None, optional
        Function to use for frequency-time-domain transformation.
        If None, np.fft.irfft will be used.
    period_length : int or None, optional
        Length of period.
    lower_bound : float or None, optional
        Smallest desired value of trans.

    Returns
    -------
    trans : 1d array with len = len(doys)

    """
    if ifft_func is None:
        ifft_func = np.fft.irfft
    if period_length is None:
        period_length = int(round(old_div(365., (doys[1] - doys[0]))))

    # the parameters do not only be as big as doys, they also have to
    # have the right number of periods!
    # fft_pars consist of the first order important parameters, the rest have
    # to be zero-padded
    fft_pars_pad = np.zeros(len(doys), dtype=complex)
    if fft_pars.dtype == complex:
        order = fft_pars.size
        fft_pars_pad[:order] = fft_pars
    else:
        order = old_div(len(fft_pars), 2)
        fft_pars_pad[:order].real = fft_pars[:order]
        fft_pars_pad[:order].imag = fft_pars[order:]

    n_periods = int(np.ceil(old_div((len(doys) + doys[0]), 365.)))
    trans = np.array(n_periods *
                     [ifft_func(fft_pars_pad, period_length)]).ravel()
    start_i = int(old_div(doys[0], period_length))
    trans = trans[start_i:start_i + len(doys)]
    if lower_bound is not None:
        trans[trans < lower_bound] = lower_bound
    return trans


class SeasonalHourlyKDE(SeasonalKDE, seasonal.Torus):

    """Estimates kernel densities for time series exhibiting seasonalities
    in daily cycles."""

    def __init__(self, data, dtimes, solution=None, doy_width=5,
               hour_neighbors=4, *args, **kwds):
        """see SeasonalKDE.__init__"""
        seasonal.Torus.__init__(self, hour_neighbors)
        super(SeasonalHourlyKDE, self).__init__(data, dtimes,
                                                doy_width=doy_width,
                                                kill_leap=True,
                                                *args, **kwds)

        self.smooth_len = hour_neighbors * 5
        if solution is not None:
            self.solution = solution
        # 1. make a 3d array with (hours, doys, years)-shape
        # 2. stack three of those 3d arrays with a daily
        #    time-shift on top of each other so that we have three
        #    subsequent days in the vertical dimension:
        #    (hours, doys, n_years) shape
        # 3. pass subslices for each doy to a kde evaluation function
        # self.torus = self._fill_torus()

    def sum_log_density(self, kernel_width, data):
        densities = self.density_per_doy(kernel_width, data,
                                         self.hour_neighbors,
                                         leave_one_out=True)
        if NE:
            return -np.nansum(ne.evaluate("log(densities)"))
        else:
            return -np.nansum(np.log(densities))

    def density_per_doy(self, kernel_width, data, hour_width,
                        eval_data=None, leave_one_out=False):
        densities = kde.apply_2d_kernel(kernel_width, data, hour_width,
                                        eval_points=eval_data)
        if leave_one_out:
            # sets main diagonal to 0 (exploiting the fact that ravel returns
            # a view)
            densities.ravel()[::len(data) + 1] = 0
        dens = old_div(np.nansum(densities, axis=1), float(len(densities) - 1))
        return dens

    @property
    def density_grid(self):
        if self._density_grid is None:
            densities = np.empty_like(self.x_grid)
            for doy_i, doy in enumerate(self.doys_unique):
                hour_slice, doy_slice = self._torus_slice(doy)
                data = self.torus[hour_slice, doy_slice]
                kernel_hour, kernel_doy = self._unpadded_index(doy)
                kernel_width = self.kernel_widths[kernel_hour, kernel_doy]
                x = self.x_grid[doy_i]
                densities[doy_i] = \
                    self.density_per_doy(kernel_width, data,
                                         self.hour_neighbors, eval_data=x)
            self._density_grid = densities
            # HACK the following will normalize the densities
            self.quantile_grid
        return self._density_grid

    @property
    def data_bounds(self):
        # hack to get daily cycles in lower_bound and upper_bound
        # self.smooth_len = 1
        data_unpad = self._unpad_torus(self.torus)
        return np.array((np.nanmin(data_unpad, axis=2).T.ravel(),
                         np.nanmax(data_unpad, axis=2).T.ravel())).T

    def lower_bound(self, doy):
        if self._lower_bound is None:
            data_unpad = self._unpad_torus(self.torus)
            mins = np.nanmin(data_unpad, axis=2)
            # smooth on an hourly basis
            mins_smooth = \
                np.array([smooth(row, self.smooth_len, periodic=True)
                          for row in mins])
            # we have to lower the smoothed mins below the actual data mins
            max_diff = np.max(mins_smooth - mins)
            diff_thresh = self.scotts_rule()
            self._lower_bound = mins_smooth - (max_diff + 5 * diff_thresh)
        hour_index, doy_index = self._unpadded_index(doy)
        return self._lower_bound[hour_index, doy_index]

    def upper_bound(self, doy):
        if self._upper_bound is None:
            data_unpad = self._unpad_torus(self.torus)
            maxs = np.nanmax(data_unpad, axis=2)
            # smooth on an hourly basis
            maxs_smooth = \
                np.array([smooth(row, self.smooth_len, periodic=True)
                          for row in maxs])
            # we have to upper the smoothed maxs below the actual data maxs
            max_diff = np.max(maxs_smooth - maxs)
            diff_thresh = self.scotts_rule()
            self._upper_bound = maxs_smooth + max_diff + 5 * diff_thresh
        hour_index, doy_index = self._unpadded_index(doy)
        return self._upper_bound[hour_index, doy_index]

    def fit(self, silverman=False, order=4):
        """
        Parameter
        ---------
        silverman : boolean or callable, optional
            Use rule of silverman instead of leave-one-out maximum
            likelihood estimation.
            Callable is executed with (n_data, n_dim).
        """
        hpd = self.hours_per_day

        if silverman:
            if not callable(silverman):
                silverman = self.silvermans_rule
            kernel_width = silverman()
            self.kernel_widths = np.full((hpd, 365), kernel_width)
            return self.kernel_widths, self.quantile_grid

        # evaluate kernel widths in the doy-hour domain
        kernel_widths = np.empty((hpd, 365))
        x0 = self.scotts_rule()
        for hour in tqdm(range(hpd), disable=(not self.verbose)):
            hour_torus = hour + self.hour_neighbors
            hour_slice = slice(hour_torus - self.hour_neighbors,
                               hour_torus + self.hour_neighbors + 1)
            for doy in range(365):
                doy_torus = doy + self.doy_width
                doy_slice = slice(doy_torus - self.doy_width,
                                  doy_torus + self.doy_width + 1)
                data = self.torus[hour_slice, doy_slice]
                kernel_width = \
                    optimize.minimize(
                        self.sum_log_density, x0,
                        args=(data,),
                        # options={"disp": True}
                    ).x
                kernel_widths[hour, doy] = kernel_width
                x0 = kernel_width
        self.kernel_width = self._kernel_widths = kernel_widths
        # self.kernel_widths = self.smooth_torus(kernel_widths, order=order)
        return self.kernel_widths, self.quantile_grid

    @property
    def solution(self):
        return self.kernel_widths, self.quantile_grid

    @solution.setter
    def solution(self, solution):
        self.kernel_widths, self.quantile_grid = solution

    def scotts_rule(self, n_data=None, n_dim=3):
        """Scott's rule of thumb for kernel bandwidth."""
        if n_data is None:
            # for getting a first guess we need the number of data in
            # each individual kde
            n_data = ((self.hour_neighbors * 2 + 1) *
                      (self.doy_width * 2 + 1) *
                      self.n_years)
        return n_data ** (-1. / (n_dim + 4))

    def pdf(self, solution, x, doys):
        if solution is not None:
            self.solution = solution
        if x is None:
            x = self.data
        if doys is None:
            doys = self.doys
        x, doys = np.atleast_1d(x, doys)
        densities = np.empty((len(doys), len(x)))
        for ii, doy in enumerate(doys):
            hour_sl, doy_sl = self._torus_index(doy)
            hour_i, doy_i = self._torus_index(doy)
            kernel_width = self.kernel_widths[hour_i, doy_i]
            densities[ii] = self.density_per_doy(kernel_width,
                                                 self.data[hour_sl, doy_sl],
                                                 eval_data=x)
        return densities


if __name__ == "__main__":
    import vg
    my_vg = vg.VG(("theta", "Qsw", "ILWR", "rh", "u", "v"), verbose=False)
    
    # # vg.vg_base.conf = vg.config_konstanz
    # met_vg = vg.VG(("theta", "ILWR", "rh", "u", "v"))
    # # from vg.meteo import avrwind
    # # data = avrwind.component2angle(met_vg.met["u"], met_vg.met["v"])[1]
    # data = met_vg.met["theta"][:3 * 8760]
    # dtimes = met_vg.times_orig[:3 * 8760]
    # sea_kde = SeasonalHourlyKDE(data, dtimes, doy_width=10, verbose=True)
    # solution = kernel_widths, quantile_grid = sea_kde.fit()

    # import vg, config_konstanz
    # vg.conf = config_konstanz
    # dtimes_, met = vg.read_met()
    # dist = SeasonalHourlyKDE(met["Qsw"], dtimes_, verbose=True)
    # dist.fit()
    # dist.scatter_pdf(None)
    # plt.show()
#    quantiles = dist.cdf(solution, x=u, doys=vg.times.datetime2doy(dtimes))
