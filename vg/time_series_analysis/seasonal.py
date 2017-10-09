"""This provides common ground for seasonal_distributions and seasonal_kde."""
from __future__ import division

from builtins import range
from builtins import object
import calendar
import numpy as np
from vg import times
from vg import helpers as my


class Seasonal(object):

    def __init__(self, data, datetimes, kill_leap=False):
        finite_mask = np.isfinite(data)
        self.data = data[finite_mask]
        self.datetimes = datetimes[finite_mask]
        self.timestep = ((self.datetimes[1] -
                          self.datetimes[0]).total_seconds() //
                         (60 ** 2 * 24))
        self.doys = times.datetime2doy(datetimes[finite_mask])
        self.doys_unique = np.unique(my.round_to_float(self.doys,
                                                       self.timestep))
        # we could have timestamps like "2004-01-01T00:30:00", so we
        # might need an offset for the unique doys
        doy0_diff = self.doys[0] - self.doys_unique[0]
        self.doys_unique += doy0_diff
        self.dt = self.doys_unique[1] - self.doys_unique[0]
        self.n_doys = len(self.doys_unique)
        if kill_leap:
            # get rid off feb 29
            self._feb29()

    def _feb29(self):
        # get rid off feb 29
        feb29_mask = ~times.feb29_mask(self.datetimes)
        self.datetimes = self.datetimes[feb29_mask]
        self.doys = times.datetime2doy(self.datetimes)
        self.data = self.data[feb29_mask]
        # doys should be from 0-365
        isleap = np.array([calendar.isleap(dt.year) for dt in self.datetimes])
        self.doys[isleap & (self.doys >= 31 + 29)] -= 1
        self.doys_unique = np.unique(my.round_to_float(self.doys,
                                                       self.timestep))
        self.n_doys = len(self.doys_unique)

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
        return int(self.timestep ** -1)


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

        Parameter
        ---------
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

    def torus_fft(self, values, hours=None, doys=None, years=None,
                  fft_order=5, padded=False):
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
        torus = np.vstack((np.roll(torus[-self.hour_neighbors:], -1, axis=1),
                           torus,
                           np.roll(torus[:self.hour_neighbors], 1, axis=1)))
        torus = np.hstack((torus[:, -self.doy_width:],
                           torus,
                           torus[:, :self.doy_width]))
        return torus

    def _unpad_torus(self, torus):
        return torus[self.hour_neighbors:-self.hour_neighbors,
                     self.doy_width:-self.doy_width]

    def _unpadded_index(self, doy):
        doy = np.atleast_1d(doy)
        hour_index = ((doy - doy.astype(int)) * self.hours_per_day).astype(int)
        doy_index = doy.astype(int) - 1
        return np.squeeze(hour_index), np.squeeze(doy_index)

    def _torus_index(self, doy):
        """Returns the index corresponding to a decimal doy."""
        hour_index, doy_index = self._unpadded_index(doy)
        return (hour_index + self.hour_neighbors,
                doy_index + self.doy_width)

    def _torus_slice(self, doy):
        """Returns slice of the torus centered around doy."""
        hour_index, doy_index = self._torus_index(doy)
        hour_slice = slice(hour_index - self.hour_neighbors,
                           hour_index + self.hour_neighbors + 1)
        doy_slice = slice(doy_index - self.doy_width,
                          doy_index + self.doy_width + 1)
        return hour_slice, doy_slice

    @property
    def doy_hour_weights(self):
        """To be used as a kernel to weight distances in the doy-hour domain.
        """
        if self._doy_hour_weights is None:
            hour_slice, doy_slice = self._torus_slice(0)
            n_hours = hour_slice.stop - hour_slice.start
            n_doys = doy_slice.stop - doy_slice.start
            # distance in the two temporal dimensions
            hour_dist, doy_dist = np.meshgrid(list(range(n_doys)),
                                              list(range(n_hours)))
            hour_middle = n_hours // 2
            doy_middle = n_doys // 2
            time_distances = np.empty((n_hours, n_doys, self.n_years))
            temp = np.sqrt((hour_dist - hour_middle) ** 2 +
                           (doy_dist - doy_middle) ** 2)
            time_distances[:] = temp[..., None]
            self._doy_hour_weights = time_distances
        return self._doy_hour_weights.ravel()
