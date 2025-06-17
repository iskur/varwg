"""This provides common ground for seasonal_distributions and seasonal_kde."""

import calendar
import hashlib
import numpy as np
import scipy.optimize as sp_optimize
from vg import times
from vg import helpers as my
from vg.smoothing import smooth
from vg.time_series_analysis import phase_randomization, distributions


def build_doy_mask(doys, doy_width, doys_unique=None):
    """Mask to access doy neighborhood in data."""
    if doys_unique is None:
        doys_unique = np.unique(doys)
    doy_mask = np.empty((len(doys_unique), len(doys)), dtype=bool)
    for doy_i, doy in enumerate(doys_unique):
        ii = (doys > doy - doy_width) & (doys <= doy + doy_width)
        if (doy - doy_width) < 0:
            ii |= doys > (365.0 - doy_width + doy)
        if (doy + doy_width) > 365:
            ii |= doys < (doy + doy_width - 365.0)
        doy_mask[doy_i] = ii
    # # take into account whether doys do not start at 1
    # doy_mask = np.roll(doy_mask, -doys.argmin())
    return doy_mask


class Seasonal(object):

    def __init__(self, data, datetimes, kill_leap=False):
        finite_mask = np.isfinite(data)
        self.data = data[finite_mask]
        self.datetimes = datetimes[finite_mask]
        # self.timestep = ((self.datetimes[1] -
        #                   self.datetimes[0]).total_seconds() //
        #                  (60 ** 2 * 24))
        self.timestep = (
            self.datetimes[1] - self.datetimes[0]
        ).total_seconds() / (60**2 * 24)
        assert self.timestep > 0
        self.doys = times.datetime2doy(datetimes[finite_mask])
        self.doys_unique = np.unique(
            my.round_to_float(self.doys, self.timestep)
        )
        # we could have timestamps like "2004-01-01T00:30:00", so we
        # might need an offset for the unique doys
        doy0_diff = self.doys.min() - self.doys_unique[0]
        self.doys_unique += doy0_diff
        assert np.max(self.doys_unique) <= 367
        self.dt = self.doys_unique[1] - self.doys_unique[0]
        self.n_doys = len(self.doys_unique)
        if kill_leap:
            # get rid off feb 29
            self._feb29()
        self._qq_shift_cache = None

    def _feb29(self):
        # get rid off feb 29
        feb29_mask = ~times.feb29_mask(self.datetimes)
        self.datetimes = self.datetimes[feb29_mask]
        self.doys = times.datetime2doy(self.datetimes)
        self.data = self.data[feb29_mask]
        # doys should be from 0-365
        isleap = np.array([calendar.isleap(dt.year) for dt in self.datetimes])
        self.doys[isleap & (self.doys >= 31 + 29)] -= 1
        self.doys_unique = np.unique(
            my.round_to_float(self.doys, self.timestep)
        )
        self.n_doys = len(self.doys_unique)

    def duplicate_feb28(self, doys):
        """Put doys in the range [1, 365] by duplicating February 28th and
        decrementing all following doys of the year.
        """
        doys_new = []
        doys_splitted = np.split(doys, np.where(doys.astype(int) == 1)[0])
        # the first element is empty when we split on doy==1 if first
        # element of doys is 1!
        if len(doys_splitted[0]) == 0:
            doys_splitted = doys_splitted[1:]
        for doys_one_year in doys_splitted:
            if np.isclose(doys_one_year[-1], 366):
                doys_one_year[doys_one_year >= 29] -= 1
            doys_new += [doys_one_year]
        doys_new = np.concatenate(doys_new)
        assert len(doys_new) == len(doys)
        return doys_new

    def doys2doys_ii(self, doys):
        """Doys indices from doy values."""
        doys = np.atleast_1d(doys)
        # use the parameters of 28. feb for 29.feb
        year_end_ii = np.where(np.diff(doys) < 0)[0] + 1
        year_end_ii = np.concatenate(([0], year_end_ii, [len(doys)]))
        for start_i, end_i in zip(year_end_ii[:-1], year_end_ii[1:]):
            year_slice = slice(start_i, end_i)
            year_doys = doys[year_slice]
            if np.max(year_doys) >= 366:
                year_doys[year_doys > 31 + 29] -= 1
                doys[year_slice] = year_doys
        doys_ii = ((doys - 1) / self.dt).astype(int)
        if len(doys_ii) < len(doys):
            doys_ii = [my.val2ind(self.doys_unique, doy) for doy in doys]
        return doys_ii

    def qq_shift(self, theta_incr, trig_pars, x=None, doys=None, **kwds):
        """Empirical estimation of shift in std-normal for given theta_incr."""
        # executing this method is expensive and in the context of
        # weathercop's conditional simulation, might happen often with
        # the same theta_incr, so do some caching here to return known
        # results
        if self._qq_shift_cache is None:
            self._qq_shift_cache = {}
        # theta_incr_key = hash(round(theta_incr[0], 6))
        theta_incr_key = hashlib.md5(
            str(round(theta_incr[0], 6)).encode()
        ).hexdigest()
        if theta_incr_key not in self._qq_shift_cache:
            qq = self.cdf(trig_pars, x=x, doys=doys, **kwds)
            zero = 1e-6
            one = 1 - zero
            stdn = distributions.norm.ppf(
                np.minimum(np.maximum(qq, zero), one)
            )
            stdn = phase_randomization.randomize2d(np.array([stdn])).squeeze()
            data = np.atleast_1d(self.data if x is None else x)
            data_mean = data.mean()

            def c2incr(c):
                xx_act = self.ppf(
                    trig_pars,
                    quantiles=distributions.norm.cdf(stdn + c),
                    doys=doys,
                    kwds=kwds,
                )
                return (xx_act.mean() - data_mean - theta_incr) ** 2

            # if self.verbose:
            #     print(f"\tFilling shift-cache for {theta_incr=}")
            result = sp_optimize.minimize_scalar(c2incr)

            # fig, ax = plt.subplots(nrows=1, ncols=1)
            # xx_act = self.ppf(trig_pars,
            #                   quantiles=distributions.norm.cdf(stdn + result.x),
            #                   doys=doys,
            #                   kwds=kwds)
            # ax.axvline(data_mean, label="data_mean", color="k")
            # ax.axvline(xx_act.mean(), label="xx_act", color="b")
            # ax.axvline(data_mean + theta_incr, label="data_mean + theta_incr",
            #            color="k", linestyle="--")
            # ax.legend()
            # plt.show()

            self._qq_shift_cache[theta_incr_key] = result.x
        return self._qq_shift_cache[theta_incr_key]

    @property
    def n_years(self):
        return self.datetimes[-1].year - self.datetimes[0].year + 1

    @property
    def hours(self):
        # general reminder: self.doys are floats, they are just a
        # representation of the date with all information except the year
        # self.hours are just the hours, and no information about minutes or
        # seconds are given
        return times.datetime2hour(self.datetimes)

    @property
    def hours_per_day(self):
        return int(self.timestep**-1)

    def _set_monthly_ticks(self, ax):
        ax.set_xticks((1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335))
        ax.set_xticklabels(
            (
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ),
            rotation=45,
        )

    def rain_probs(self, threshold, doys=None):
        if doys is None:
            doys = self.doys
        probs_per_doy = np.zeros_like(self.doys_unique)
        for doy_i in self.doys_unique.astype(int) - 1:
            data = self.data[self.doy_mask[doy_i]]
            probs_per_doy[doy_i] = np.mean(data > threshold)
        probs_per_doy = smooth(probs_per_doy, self.doy_width, periodic=True)
        return probs_per_doy[doys.astype(int) - 1]


class Torus(Seasonal):

    def __init__(self, hour_neighbors):
        self.hour_neighbors = hour_neighbors
        # for property caching
        self._torus = None

    @property
    def torus(self):
        if self._torus is None:
            self._torus = self._construct_torus(self.data)
        return self._torus

    def _construct_torus(self, values, hours=None, doys=None, years=None):
        """Returns a 3d array representation of the 1d input.

        Parameters
        ----------
        values : 1d array
        """
        # hours, doys and years are constructed so that they can be
        # used as indices for the torus
        if hours is None:
            hours = self.hours
        if doys is None:
            doys = self.doys
        if years is None:
            first_year = self.datetimes[0].year
            years = [dt.year - first_year for dt in self.datetimes]
            n_years = self.n_years
        else:
            n_years = len(np.unique(years))
        hours = list(hours.astype(int))
        doys = list(doys.astype(int) - 1)
        years = list(years)

        # we deleted the 29 of february to have a sane and full array
        # with 365 days in the day dimension
        torus = np.full((self.hours_per_day, 365, n_years), np.nan)
        torus[[hours, doys, years]] = values
        torus = self._pad_torus(torus)

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(len(np.unique(years)))
        # for year_i, ax in enumerate(np.ravel(axs)):
        #     # , cmap=plt.get_cmap("BuPu"))
        #     cm = ax.matshow(torus[..., year_i])
        # # plt.colorbar(cm)
        # plt.show()
        return torus

    def torus_fft(
        self,
        values,
        hours=None,
        doys=None,
        years=None,
        fft_order=5,
        padded=False,
    ):
        if np.ndim(values) == 1:
            values = self._construct_torus(values, hours, doys, years)
        if not padded:
            values = self._pad_torus(values)
        omega = np.fft.fft2(values)
        # # avoid long hourly frequencies
        # omega[24:-24] = 0
        if fft_order is not None:
            nth_largest = np.sort(np.ravel(np.abs(omega)))[-fft_order]
            omega[np.abs(omega) < nth_largest] = 0
        return np.squeeze(omega)

    def smooth_torus(self, values, fft_order=5, padded=False):
        omega = self.torus_fft(values, fft_order=5, padded=padded)
        return np.fft.irfft2(omega, s=values.shape)

    def _pad_torus(self, torus):
        # to achieve periodicity, i.e. move up -> advance in hours,
        # move right -> advance in days
        torus = np.vstack(
            (
                np.roll(torus[-self.hour_neighbors :], -1, axis=1),
                torus,
                np.roll(torus[: self.hour_neighbors], 1, axis=1),
            )
        )
        torus = np.hstack(
            (torus[:, -self.doy_width :], torus, torus[:, : self.doy_width])
        )
        return torus

    def _unpad_torus(self, torus):
        return torus[
            self.hour_neighbors : -self.hour_neighbors,
            self.doy_width : -self.doy_width,
        ]

    def _unpadded_index(self, doy):
        doy = np.atleast_1d(doy)
        hour_index = ((doy - doy.astype(int)) * self.hours_per_day).astype(int)
        doy_index = doy.astype(int) - 1
        return np.squeeze(hour_index), np.squeeze(doy_index)

    def _torus_index(self, doy):
        """Returns the index corresponding to a decimal doy."""
        hour_index, doy_index = self._unpadded_index(doy)
        return (hour_index + self.hour_neighbors, doy_index + self.doy_width)

    def _torus_slice(self, doy):
        """Returns slice of the torus centered around doy."""
        hour_index, doy_index = self._torus_index(doy)
        hour_slice = slice(
            hour_index - self.hour_neighbors,
            hour_index + self.hour_neighbors + 1,
        )
        doy_slice = slice(
            doy_index - self.doy_width, doy_index + self.doy_width + 1
        )
        return hour_slice, doy_slice

    @property
    def doy_hour_weights(self):
        """To be used as a kernel to weight distances in the doy-hour domain."""
        if self._doy_hour_weights is None:
            hour_slice, doy_slice = self._torus_slice(0)
            n_hours = hour_slice.stop - hour_slice.start
            n_doys = doy_slice.stop - doy_slice.start
            # distance in the two temporal dimensions
            hour_dist, doy_dist = np.meshgrid(
                list(range(n_doys)), list(range(n_hours))
            )
            hour_middle = n_hours // 2
            doy_middle = n_doys // 2
            time_distances = np.empty((n_hours, n_doys, self.n_years))
            temp = np.sqrt(
                (hour_dist - hour_middle) ** 2 + (doy_dist - doy_middle) ** 2
            )
            time_distances[:] = temp[..., None]
            self._doy_hour_weights = time_distances
        return self._doy_hour_weights.ravel()
