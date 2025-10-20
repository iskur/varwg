import copy
import datetime
import os
import re
import shlex
import sys
import warnings
from pickle import UnpicklingError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from tqdm import tqdm

import varwg
import varwg.time_series_analysis.seasonal_kde as skde
from varwg import helpers as my, shelve
from varwg.meteo import avrwind, meteox2y
from varwg.time_series_analysis import (
    distributions,
    models,
)
from varwg.time_series_analysis import (
    seasonal_distributions as sd,
)

try:
    from varwg import config as conf
except ImportError:
    from varwg import config_template as conf

    conf_filepath = conf.__file__
    if conf_filepath.endswith(".pyc"):
        conf_filepath = conf_filepath[:-1]
    warnings.warn(
        'Could not import "config.py".\n'
        + 'Edit "%s" and rename it to "config.py"' % conf_filepath
    )


PY2 = sys.version_info.major == 2
cache_filename = "seasonal_solutions_{version}.sh".format(
    version="py2" if PY2 else "py3"
)


def detrend(values):
    """Detrend by subtracting the linear regression function."""
    dummy_time = np.arange(len(values))
    b, a = stats.linregress(dummy_time, values)[:2]
    return values + (values.mean() - (a + b * dummy_time))


def daily_wind(met, sum_interval=24):
    """Aggregates daily wind direction and speed from hourly values."""
    u_daily = my.sumup(met["u"], sum_interval, sum_to_nan=True)
    v_daily = my.sumup(met["v"], sum_interval, sum_to_nan=True)
    return avrwind.component2angle(u_daily, v_daily)


def met_as_array(met, p=0, T=None, var_names=None):
    """Returns the met - dictionary as an array. Rows are sorted
    alpha-numerically according to the variable names."""
    var_names = list(met.keys()) if var_names is None else var_names
    K = len(var_names)
    if T is None:
        T = len(met[var_names[0]]) - p
    # this lets some outer code access the following variables without
    # returning them explicitly
    met_as_array.var_names, met_as_array.T = var_names, T
    # the "T + p" is there to include the presample timesteps.
    data = np.nan * np.empty((K, T + p))
    for var_name in var_names:
        data[var_names.index(var_name)] = met[var_name]
    return data


def met2array(filepath, met_kwds=None, as_array_kwds=None, sumup_kwds=None):
    met_kwds, as_array_kwds, sumup_kwds = [
        {} if x is None else x for x in (met_kwds, as_array_kwds, sumup_kwds)
    ]
    dt_hourly, met = read_met(filepath, **met_kwds)
    data_hourly = met_as_array(met, **as_array_kwds)
    data, dt = my.sumup(data_hourly, times_=dt_hourly, **sumup_kwds)
    try:
        n_sumup = sumup_kwds["width"]
    except KeyError:
        n_sumup = 24
    return data / n_sumup, dt, met_as_array.var_names


def dyresm2array(filepath, *args, **kwds):
    met_kwds = kwds.pop("met_kwds") if "met_kwds" in kwds else {}
    met_kwds["startfrom"] = 5
    return met2array(filepath, met_kwds=met_kwds, *args, **kwds)


def _parse_time(var_dict):
    """Get the time information in var_dict.
    This code is adjusted every time someone throws new data at me."""
    datetimes = None
    if "Date" in var_dict and "hour" in var_dict:
        datetimes = varwg.times.str2datetime(
            np.array(
                [
                    "%s %s" % (date_str, hour_str)
                    for date_str, hour_str in zip(
                        var_dict["Date"], var_dict["hour"]
                    )
                ]
            ),
            "%d. %m %y %H",
        )
        del var_dict["Date"]
        del var_dict["hour"]
    if "Julian" in var_dict:
        datetimes = varwg.times.cwr2datetime(var_dict["Julian"])
        del var_dict["Julian"]
    elif "YrDayNum" in var_dict:
        datetimes = varwg.times.cwr2datetime(var_dict["YrDayNum"])
        del var_dict["YrDayNum"]
    elif "jd" in var_dict:
        datetimes = varwg.times.cwr2datetime(var_dict["jd"])
        del var_dict["jd"]
    if "chron" in var_dict:
        datetimes = varwg.times.str2datetime(
            var_dict["chron"], "(%m/%d/%y %H:%M:%S)"
        )
        del var_dict["chron"]
    elif "date" in var_dict:
        datetimes = varwg.times.str2datetime(var_dict["date"])
        del var_dict["date"]
    elif "time-iso" in var_dict:
        datetimes = varwg.times.iso2datetime(var_dict["time-iso"])
        del var_dict["time"]
    elif "time" in var_dict:
        datetimes = varwg.times.str2datetime(
            var_dict["time"], "%Y-%m-%d %H:%M:%S"
        )
        del var_dict["time"]
    elif "times" in var_dict:
        datetimes = varwg.times.str2datetime(
            var_dict["times"], "%Y-%m-%d %H:%M:%S"
        )
        del var_dict["times"]
    if "year" in var_dict and "doy" in var_dict:
        datetimes = varwg.times.str2datetime(var_dict["year"], "%Y")
        datetimes += np.array(
            [datetime.timedelta(int(doy)) for doy in var_dict["doy"]]
        )
        del var_dict["year"], var_dict["doy"]
    if "Date" in var_dict and "Time" in var_dict:
        # as seen in Excel
        datetimes = varwg.times.xls2datetime(
            [int(day) for day in var_dict["Date"]]
        )
        datetimes += np.array(
            [
                datetime.timedelta(hours=int(hour.split(":")[0]))
                for hour in var_dict["Time"]
            ]
        )
        del var_dict["Date"], var_dict["Time"]
    if "ta" in var_dict and "mo" in var_dict and "jahr" in var_dict:
        datetimes = np.array(
            [
                datetime.datetime(int(year), int(month), int(day))
                for year, month, day in zip(
                    var_dict["jahr"], var_dict["mo"], var_dict["ta"]
                )
            ]
        )
        del var_dict["ta"]
        del var_dict["mo"]
        del var_dict["jahr"]
    if "index" in var_dict:
        del var_dict["index"]
    if datetimes is None:
        raise TypeError("Could not parse dates.")
    return var_dict, datetimes


# @my.pickle_cache(os.path.join(conf.cache_dir, "met_%s.pkl"))  # , warn=False)
def read_met(
    filepath=None,
    minimum_water_temp=0.1,
    delimiter="\t",
    verbose=True,
    main_diff=None,
    with_conversions=False,
    **kwds,
):
    if filepath is None:
        filepath = conf.met_file
    var_dict = my.csv2dict(filepath, delimiter=delimiter, **kwds)
    # get rid of quotes around filenames
    var_dict = {
        shlex.split(var_name)[0]: np.asarray(values)
        for var_name, values in list(var_dict.items())
    }

    var_dict, datetimes = _parse_time(var_dict)

    met = {
        key: np.array(
            [val if val != "" else np.nan for val in values], dtype=float
        )
        for key, values in list(var_dict.items())
    }

    from varwg import times

    met = {
        key: varwg.times.regularize(
            val, datetimes, nan=True, main_diff=main_diff
        )[0]
        for key, val in list(met.items())
    }
    # get also the regularized datetimes

    datetimes = varwg.times.regularize(
        np.empty(len(datetimes)), datetimes, main_diff=main_diff
    )[1]

    if verbose:
        n_nan = np.sum([np.isnan(val) for val in list(met.values())])
        print("\tFound %d missing or nan values" % n_nan)
        for var_name, val in sorted(met.items()):
            gaps = my.gaps(val)
            if len(gaps) > 0:
                n_gaps = np.sum(np.isnan(val))
                gaps_perc = float(n_gaps) / len(val) * 100
                print("\t%s (%d, %.2f%%)" % (var_name, n_gaps, gaps_perc))
                for start, end in gaps:
                    if start == end:
                        print("\t\t", datetimes[start])
                    else:
                        print(
                            "\t\t%s - %s (%d)"
                            % (
                                datetimes[start],
                                datetimes[end],
                                end - start + 1,
                            )
                        )

    if "Cloud_Cover" not in var_dict:
        try:
            var_dict["Cloud_Cover"] = meteox2y.lw2clouds(
                np.array(var_dict["ILWR"], dtype=float),
                np.array(var_dict["theta"], dtype=float),
                e=np.array(var_dict["e"], dtype=float),
            )
        except KeyError:
            pass

    if "wdir" in met and "U" in met:
        u, v = avrwind.angle2component(met["wdir"], met["U"])
        met.update({"u": u, "v": v})

    if ("rh" not in met) and ("e" in met) and ("theta" in met):
        met["rh"] = meteox2y.vap_p2rel(met["e"], met["theta"])
        too_dry_ii = met["rh"] < 0
        n_too_dry = np.sum(too_dry_ii)
        if n_too_dry > 0:
            if verbose:
                warnings.warn(
                    (
                        "Input of e and theta caused %d rel. "
                        + "humidities to be < 0. Capping them at 0."
                    )
                    % n_too_dry
                )
            met["rh"][too_dry_ii] = np.nan
        too_moist_ii = met["rh"] > 1.01
        n_too_moist = np.sum(too_moist_ii)
        if n_too_moist > 0:
            if verbose:
                warnings.warn(
                    (
                        "Input of e and theta caused %d rel. "
                        + "humidities to be > 1. Capping them at 1."
                    )
                    % n_too_moist
                )
            met["rh"][too_moist_ii] = 1.01  # np.nan
    if "rh" in met and np.nanmax(met["rh"]) > 50:
        met["rh"] /= 100.0
    for var_name in list(met.keys()):
        if var_name.startswith("wtemp"):
            too_cold_ii = met[var_name] < 0
            n_too_cold = np.sum(too_cold_ii)
            if n_too_cold > 0:
                if verbose:
                    warnings.warn(
                        (
                            "%d temperatures < 0 in %s. Capping them "
                            + "at %.3f"
                        )
                        % (n_too_cold, var_name, minimum_water_temp)
                    )
                met[var_name][too_cold_ii] = minimum_water_temp

    if with_conversions:
        var_names = list(met.keys())
        for conversion in list(conf.conversions):
            times, data, var_names = conversion(
                datetimes,
                np.array([met[var_name] for var_name in var_names]),
                var_names,
                inverse=True,
            )
        met = {
            var_name: data[var_names.index(var_name)] for var_name in var_names
        }
    return datetimes, met


class VGBase(object):
    """Handle everything except plotting."""

    def __init__(
        self,
        var_names,
        met_file=None,
        sum_interval=24,
        max_nans=12,
        plot=False,
        separator="\t",
        refit=None,
        detrend_vars=None,
        verbose=False,
        data_dir=None,
        cache_dir=None,
        dump_data=True,
        non_rain=None,
        rain_method=None,
        neg_rain_doy_width=30,
        neg_rain_fft_order=2,
        neg_kwds=None,
        infill=False,
        fit_kwds=None,
        **met_kwds,
    ):
        # external_var_names=None,
        # external_var_names : sequence of str, optional
        #     Must be present in the met-file.
        self.var_names = var_names
        self.verbose = verbose
        self.met_file = conf.met_file if met_file is None else met_file
        self.data_dir = conf.data_dir if data_dir is None else data_dir
        self.fit_kwds = {} if fit_kwds is None else fit_kwds
        if cache_dir is not None:
            self.cache_dir = cache_dir
            self.seasonal_cache_file = os.path.join(cache_dir, cache_filename)
        else:
            self.cache_dir = conf.cache_dir
            self.seasonal_cache_file = conf.seasonal_cache_file
            if not re.match(r".*py[2,3]\.sh$", self.seasonal_cache_file):
                # py2/3 are incompatible here, so we write different cache
                # files depending on the python version
                name_parts = self.seasonal_cache_file.rsplit(".")
                self.seasonal_cache_file = "%s_%s.%s" % (
                    name_parts[0],
                    "py2" if PY2 else "py3",
                    ".".join(name_parts[1:]),
                )
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.dump = dump_data
        # will be used in _load_and_prepare_data
        self.met = self.times_orig = None
        # are we given a sequence? then store sum_interval in a way, that
        # data can be divided by it
        try:
            self.sum_interval = np.array(sum_interval)[:, np.newaxis]
        except IndexError:
            # let's make this the same shape, so life gets easier later on
            self.sum_interval = np.array(self.K * [sum_interval])[
                :, np.newaxis
            ]
        self.plot = plot

        # attributes that will be set with meaningfull values in fit()
        # AM is also used to store the parameters of a pure VAR-process
        self.AM = self.sigma_u = self.residuals = self.means = None
        self.p = self.q = None

        # these attributes will be populated in simulate() and disaggregate()
        self.T_sim = self.mean_arrival = self.disturbance_std = None
        self.theta_incr = self.theta_grad = self.fixed_variables = None
        self.primary_var = self.climate_signal = self.start_date = None
        self.m_t = self.sim = self.sim_sea = self.sim_times = None
        self.outfilepath = self.sim_sea_dis = self.dis_times = None
        self.var_names_dis = self.ex_in = self.ex_out = None
        self.svar_doy_width = self.svar_fft_order = None

        # cached attributes
        self._data_doys = self._data_doys_raw = None
        self._sim_doys = self._dis_doys = None
        self._sim_doys_len = self._dis_doys_len = np.nan
        self._cov_trans = None

        if self.verbose:
            print("Loading input data.")
        (self.times, self.data_raw) = self._load_and_prepare_data(
            separator, max_nans=max_nans, **met_kwds
        )

        if detrend_vars:
            for var_name in detrend_vars:
                var_ii = var_names.index(var_name)
                self.data_raw[var_ii] = detrend(self.data_raw[var_ii])

        if self.seasonal_fitting:
            if self.verbose:
                print("Fitting seasonal distributions.")
            self.data_trans, self.dist_sol = self._fit_seasonal(refit)

        # if "R" in self.var_names:
        #     if self.verbose:
        #         print(f"Transforming 'negative rain' (using {rain_method})")
        #     # populated in negative_rain
        #     self.rain_mask = np.ones(self.T_summed, dtype=bool)
        #     rain_i = self.var_names.index("R")
        #     rain = self.data_raw[rain_i]
        #     self.data_trans = self._negative_rain(
        #         self.data_trans,
        #         rain,
        #         self.data_doys,
        #         doy_width=neg_rain_doy_width,
        #         fft_order=neg_rain_fft_order,
        #         var_names=non_rain,
        #         method=rain_method,
        #         kwds=neg_kwds,
        #     )

        for var_name, (dist, sol) in self.dist_sol.items():
            if isinstance(dist, skde.SeasonalKDE):
                continue
            if not isinstance(dist.dist, distributions.RainMix):
                continue
            if self.verbose:
                print(
                    f"Transforming zero-values of {var_name} (using {rain_method})"
                )
            # populated in negative_rain
            self.rain_mask = np.ones(self.T_summed, dtype=bool)
            rain_i = self.var_names.index(var_name)
            rain = self.data_raw[rain_i]
            self.data_trans = self._negative_rain(
                self.data_trans,
                rain,
                self.data_doys,
                doy_width=neg_rain_doy_width,
                fft_order=neg_rain_fft_order,
                var_names=non_rain,
                method=rain_method,
                kwds=neg_kwds,
                self_name=var_name,
            )

        if infill:
            self.data_trans = self.infill_trans_nans()

    @property
    def sum_interval_dict(self):
        """Maps var_name to sum_interval."""
        return dict(
            (var_name, self.sum_interval[self.var_names.index(var_name)])
            for var_name in self.var_names
        )

    @property
    def primary_var_ii(self):
        """The row index of the primary variable."""
        if self.primary_var is None:
            raise RuntimeError(
                "primary_var is not set. Have you called simulate yet?"
            )
        try:
            return [
                self.var_names.index(prim_var) for prim_var in self.primary_var
            ]
        except ValueError:
            warnings.warn("No %s in input." % self.primary_var)

    @property
    def output_resolution(self):
        """In hours."""
        delta = self.times[1] - self.times[0]
        # there is timedelta.total_seconds in python 2.7, but lets be kind to
        # the conservative people -- oh, thank you ;-)
        return (
            24 * delta.days
            + (delta.seconds + 1e-3 * delta.microseconds) / 60**2
        )

    @property
    def K(self):
        """The number of variables."""
        return len(self.var_names)

    @property
    def T_data(self):
        """Length of the input data time series ('hourly')."""
        return len(self.times_orig)

    @property
    def T_summed(self):
        """Length of aggregated input data ('daily')."""
        return len(self.times)

    @property
    def data_doys(self):
        """Days of year of aggregated input data."""
        if self._data_doys is None:
            self._data_doys = varwg.times.datetime2doy(self.times)
        return self._data_doys

    @property
    def data_doys_raw(self):
        """Days of year of raw (hourly) input data."""
        if self._data_doys_raw is None:
            self._data_doys_raw = varwg.times.datetime2doy(self.times_orig)
        return self._data_doys_raw

    @property
    def sim_doys(self):
        """Days of year of aggregated output data."""
        if self._sim_doys is None or len(self.sim_times) != self._sim_doys_len:
            self._sim_doys = varwg.times.datetime2doy(self.sim_times)
            self._sim_doys_len = len(self._sim_doys)
        return self._sim_doys

    @property
    def dis_doys(self):
        """Days of year of disaggregated output data."""
        if self._dis_doys is None or len(self._dis_doys) != self._dis_doys_len:
            self._dis_doys = varwg.times.datetime2doy(self.dis_times)
            self._dis_doys_len = len(self._dis_doys)
        return self._dis_doys

    @property
    def start_hour_of_src(self):
        return self.times_orig[0].hour

    def _diff(self, other, plot=False, verbose=True):
        """Show differences in attribute values between this and other VG instance.

        For debugging.
        """
        if not isinstance(other, type(self)):
            raise RuntimeError(
                "Can only compare two VG objects (got {type(other)})."
            )
        diff = my.recursive_diff("", self, other, verbose=verbose, plot=plot)
        if diff:
            # the first dictionary is uninformative and has only one element
            diff = diff.popitem()[1]
        if plot and my.recursive_diff.fig_axs is not None:
            # my.recursive_diff does not know about the variables' names. but we do.
            for name, (fig, axs) in my.recursive_diff.fig_axs.items():
                if len(axs) == self.K:
                    for ax, var_name in zip(axs, self.var_names):
                        ax[0].set_title(var_name)
            plt.show()
        my.recursive_diff.clear_cache()
        return diff

    def fitted_medians(self, var_name, doys=None):
        """Medians of the fitted seasonal distribution.

        Usefull to construct `climate_signal`s

        Parameters
        ----------
        var_name : str or sequence of strings
        doys : 1d array
            If None, `sim_times` is used. If that is not available,
            `data_doys`.

        """
        if isinstance(var_name, str):
            var_names = (var_name,)
        else:
            var_names = var_name
        medians = []
        for var_name in var_names:
            dist, solution = self.dist_sol[var_name]
            var_i = self.var_names.index(var_name)
            if doys is None:
                if self.sim_times is None:
                    doys = self.data_doys
                else:
                    doys = self.sim_doys
            medians += [
                (
                    dist.ppf(solution, np.full_like(doys, 0.5), doys)
                    / self.sum_interval[var_i]
                )
            ]
        return np.squeeze(medians)

    def _shuffle(
        self,
        nn,
        m,
        tpd=24,
        autocorr_len=48,
        doys_in=None,
        doys_out=None,
        doy_tolerance=15,
        nan_mask=None,
    ):
        """Draw a clustered sample of size m with elements [0,nn]."""

        def mod0(x, y):
            # like a normal modulus, but knows how to divide by 0!!1
            return x % y if y != 0 else 0

        seasonal = doys_in is not None and doys_out is not None
        if seasonal:
            pool0 = np.where(
                varwg.times.doy_distance(doys_out[0], doys_in) <= doy_tolerance
            )[0]
            # pool_len = len(pool0)

        if nan_mask is not None:
            finite_ii = np.where(~nan_mask)[0]
            # shrink pool
            pool0 = pool0[
                (pool0 >= (finite_ii[0] - mod0(0, tpd)))
                & (pool0 <= (finite_ii[-1] - mod0(0, tpd)))
            ]
            finite_ii = set(finite_ii)
            # pool_len = len(pool0)
        else:
            finite_ii = set(np.arange(nn))

        def choose_chunk(dst_point):
            nan_in_output = True
            # dst_point -= self.start_hour_of_src
            # dst_point += self.start_hour_of_src
            # dst_point -= 1
            while nan_in_output:
                if seasonal:
                    # pool = (pool0 + dst_point) % nn
                    doy_out = doys_out[dst_point % m]
                    pool = np.where(
                        varwg.ctimes.doy_distance(doy_out, doys_in)
                        <= doy_tolerance
                    )[0]
                    pool = list(set(pool) & finite_ii)
                    src_point = pool[varwg.rng.integers(len(pool))]
                    while src_point >= (nn - autocorr_len):
                        src_point = pool[varwg.rng.integers(len(pool))]
                else:
                    src_point = varwg.rng.choice(finite_ii[:-autocorr_len])
                # src_point -= self.start_hour_of_src
                # hour of day from 0 to 23 in destination
                hour_of_dst = mod0(dst_point, tpd)
                # ensure the same hour of day in source
                src_point += -mod0(src_point, tpd) + hour_of_dst
                src_point = min(src_point, nn - autocorr_len)
                chunk_ii = np.arange(src_point, src_point + autocorr_len)
                # while np.any(nan_mask[chunk_ii]):
                #     import ipdb; ipdb.set_trace()
                #     if (max(chunk_ii) + tpd) > nn:
                #         break
                #     chunk_ii = np.array(chunk_ii) + tpd
                if not np.any(nan_mask[chunk_ii]):
                    nan_in_output = False
            return chunk_ii

        progress = tqdm if self.verbose else lambda x: x
        indices = np.array(
            [
                choose_chunk(dst_point)
                for dst_point in progress(
                    range(0, m + autocorr_len, autocorr_len)
                )
            ]
        )
        # indices -= self.start_hour_of_src
        return indices.ravel()[:m]

    def _gen_deltas_input(
        self, var_names_dis, tpd, longitude=None, latitude=None
    ):
        """Generate the pool of deltas_input for varwg.disaggregate."""
        if longitude is None:
            longitude = conf.longitude
        if latitude is None:
            latitude = conf.latitude
        # due to the interpolation with interp1d, we often have to skip the
        # last day (24 hours)
        # size of pool we can draw from:
        nn = self.T_data // tpd * tpd - tpd
        # these are in "hourly" discretization
        doys_in = self.data_doys_raw[:nn]
        m = self.T_sim * tpd - tpd  # no of hourly timesteps (simulated)
        sim_sea_dis = self.sim_sea.repeat(tpd).reshape(-1, m + tpd)[:, :-tpd]
        deltas_input = np.zeros((self.K, nn))
        sim_interps = np.copy(sim_sea_dis)
        for var_name in var_names_dis:
            var_i = self.var_names.index(var_name)
            # hourly measured values
            var_h = my.interp_nonfin(
                self.met[var_name][: nn + tpd], max_interp=3
            )
            # if self.start_hour_of_src:
            #     print(f"prepending var_h by {self.start_hour_of_src=} steps")
            #     var_h = np.concatenate((np.zeros(self.start_hour_of_src),
            #                             var_h))[:nn + tpd]
            # daily averages of measured values:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                var_d = np.nanmean(var_h.reshape(-1, tpd), axis=1)
            # 'time' array for interp1d
            hourly_input_times = np.arange(var_h.shape[0])
            # interpolate between daily values (measurements)
            f_interp = interpolate.interp1d(hourly_input_times[::tpd], var_d)
            var_interp = f_interp(hourly_input_times[:-tpd])
            # residuals to draw from
            deltas_input[var_i] = var_h[:-tpd] - var_interp
            # deltas_input[var_i] = var_h[tpd // 2:-tpd // 2] - var_interp
            # if the variable has lower/upper limits, store the percentage of
            # the distance covered between the interpolated line and the limit
            limits = copy.copy(conf.par_known[var_name])
            if var_name.startswith("Qsw"):

                def pot_s(doys):
                    # hourly = meteox2y_cy.pot_s_rad(
                    hourly = meteox2y.pot_s_rad(
                        doys,
                        lat=latitude,
                        longt=longitude,
                        tz_mer=None,
                    )
                    return hourly * self.sum_interval[var_i]

                limits["u"] = pot_s

                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
                # axs[0].plot(self.times_orig[:nn],
                #             var_h[:-tpd], label="var_h")
                # axs[0].plot(self.times_orig[:nn],
                #             var_interp, label="var_interp")
                # axs[0].plot(self.times_orig[:nn],
                #             pot_s(doys_in) / self.sum_interval[var_i],
                #             label="pot_s")
                # axs[0].legend(loc="best")
                # axs[1].plot(self.times_orig[:nn], deltas_input[var_i])
                # plt.show()
                # __import__('pdb').set_trace()

            pos_mask = deltas_input[var_i] > 0
            neg_mask = ~pos_mask

            # we have to interpret a threshold as a lower limit (no negative
            # rain, please)
            seas_dist, _ = self.dist_sol[var_name]
            if hasattr(seas_dist, "dist") and hasattr(
                seas_dist.dist, "thresh"
            ):
                if limits is None:
                    limits = {}
                limits["l"] = conf.array_gen(conf.threshold)
                # limits["l"] = conf.array_gen(0.)

            if limits is not None and ("l" in limits or "lc" in limits):
                lower_func = limits["l"] if "l" in limits else limits["lc"]
                lower = (
                    lower_func(doys_in[neg_mask]) / self.sum_interval[var_i]
                )
                deltas = deltas_input[var_i, neg_mask]
                interps = var_interp[neg_mask]
                div_mask = ~np.isclose(interps, lower)
                lower_perc = np.full_like(interps, 1e-12)
                lower_perc[div_mask] = deltas[div_mask] / (
                    interps[div_mask] - lower[div_mask]
                )
                lower_perc[interps < lower] = 0
                lower_perc[lower_perc < -1.0] = -1.0
                deltas_input[var_i, neg_mask] = lower_perc
            if limits is not None and ("u" in limits or "uc" in limits):
                upper_func = limits["u"] if "u" in limits else limits["uc"]
                upper = (
                    upper_func(doys_in[pos_mask]) / self.sum_interval[var_i]
                )
                deltas = deltas_input[var_i, pos_mask]
                interps = var_interp[pos_mask]
                div_mask = ~np.isclose(interps, upper)
                upper_perc = np.full_like(interps, 1 - 1e-12)
                upper_perc[div_mask] = deltas[div_mask] / (
                    upper[div_mask] - interps[div_mask]
                )
                upper_perc[interps > upper] = 0
                upper_perc[upper_perc > 1.0] = 1.0
                deltas_input[var_i, pos_mask] = upper_perc

            # not a real 'time' as above
            hourly_output_times = np.arange(self.T_sim * tpd)
            # interpolate between daily values (simulation)
            f_interp = interpolate.interp1d(
                hourly_output_times[::tpd], self.sim_sea[var_i]
            )
            sim_interps[var_i] = f_interp(hourly_output_times[:-tpd])
        return deltas_input, sim_interps

    def _add_deltas(
        self,
        deltas_input,
        sim_interps,
        var_names_dis,
        tpd,
        event_dt=None,
        factors=None,
        doy_tolerance=15,
    ):
        # these are in "hourly" discretization
        doys_in = self.data_doys_raw[:-tpd]
        if event_dt is not None:
            # the event datetimes have to be mapped into the hourly
            # discretization
            year_days_out = varwg.times.time_part(self.dis_times, "%Y%j")
            year_days_events = varwg.times.time_part(event_dt, "%Y%j")
            event_ii = np.where(year_days_out == year_days_events[:, None])[1]

        # drawing from pool: use same indices for all to preserve dependencies
        nan_mask = np.any(np.isnan(deltas_input), axis=0)
        # size of pool we can draw from:
        nn = self.T_data // tpd * tpd - tpd
        m = self.T_sim * tpd - tpd  # no of hourly timesteps (simulated)
        # TODO: Check for fitted Rain instances explicitly!
        if "R" in var_names_dis:
            deltas_drawn = np.empty((len(var_names_dis), m))
            rain_i = var_names_dis.index("R")
            thresh = conf.threshold
            hourly_rain_in = my.interp_nonfin(
                self.met["R"][: nn + tpd], max_interp=3
            )
            daily_rain_in = hourly_rain_in.reshape(-1, tpd).mean(axis=1)
            rain_mask_in = (daily_rain_in > thresh).repeat(tpd)[:nn]
            daily_rain_out = self.sim_sea[rain_i]
            rain_mask_out = (daily_rain_out > 0).repeat(tpd)[:m]

            # shuffle in wet chunks
            chosen_chunks = self._shuffle(
                nn,
                m,
                tpd,
                2 * tpd,
                doys_in,
                self.dis_doys,
                nan_mask=nan_mask | rain_mask_in,
                doy_tolerance=doy_tolerance,
            )
            chosen_chunks = chosen_chunks[rain_mask_out]
            deltas_drawn[:, rain_mask_out] = deltas_input[:, chosen_chunks]
            if event_dt is not None and factors is not None:
                deltas_drawn[:, event_ii] *= factors
            sim_sea_dis = np.where(
                rain_mask_out, sim_interps + deltas_drawn, sim_interps
            )
            sim_sea_dis[rain_i, np.isclose(deltas_drawn[rain_i], 0)] = 0

            # shuffle in dry chunks
            chosen_chunks = self._shuffle(
                nn,
                m,
                tpd,
                2 * tpd,
                doys_in,
                self.dis_doys,
                nan_mask=nan_mask | ~rain_mask_in,
                doy_tolerance=doy_tolerance,
            )
            chosen_chunks = chosen_chunks[~rain_mask_out]
            deltas_drawn[:, ~rain_mask_out] = deltas_input[:, chosen_chunks]
            if event_dt is not None and factors is not None:
                deltas_drawn[:, event_ii] *= factors
            sim_sea_dis = np.where(
                ~rain_mask_out, sim_sea_dis + deltas_drawn, sim_sea_dis
            )
            sim_sea_dis[rain_i, np.isclose(deltas_drawn[rain_i], 0)] = 0

            # TODO: check if this worsens the hourly distribution fit
            sim_sea_dis[rain_i, ~rain_mask_out] = 0
        else:
            chosen_chunks = self._shuffle(
                nn,
                m,
                tpd,
                2 * tpd,
                doys_in,
                self.dis_doys,
                nan_mask=nan_mask,
                doy_tolerance=doy_tolerance,
            )
            deltas_drawn = deltas_input[:, chosen_chunks]
            if event_dt is not None and factors is not None:
                deltas_drawn[:, event_ii] *= factors
            sim_sea_dis = sim_interps + deltas_drawn
        return deltas_drawn, sim_sea_dis

    def _load_and_prepare_data(self, delimiter="\t", max_nans=12, **met_kwds):
        """Load the data from the met_file and aggregate it according to
        sum_interval (both defined in __init__). Plus plotting if requested."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        try:
            # get the data
            # this covers the case that a non-default met filename was given
            # that exists in the data_dir but not in the working path
            if (self.met_file is not None) and (
                not os.path.exists(self.met_file)
            ):
                self.met_file = os.path.join(self.data_dir, self.met_file)
            if self.verbose:
                print(f"Reading {self.met_file}")
            self.times_orig, self.met = read_met(
                self.met_file,
                delimiter=delimiter,
                verbose=self.verbose,
                **met_kwds,
            )
        except TypeError:
            # warnings.warn("While reading met-file:\n%s" % exc)
            try:
                # could be a pandas DataFrame
                self.times_orig = self.met_file.index.to_pydatetime()
                self.met = self.met_file.to_dict("list")
            except AttributeError:
                # or just a dict
                self.times_orig = self.met_file["datetimes"]
                self.met = {
                    key: val
                    for key, val in list(self.met_file.items())
                    if key != "datetime"
                }

        # convert the dictionary into an array.
        # mind the order: alpha-numeric according to var_names
        # will be shorter once summed up
        # T_raw_data = len(self.met[self.var_names[0]])
        # self.K corresponds to all variables. the wtemp variables are
        # unfortunately stored someplace else and should be ignored here
        # so for the shape of data, we do not count to K, but something lower
        # if there are water temperatures in var_names
        var_names_part = [
            var_name
            for var_name in self.var_names
            if (not var_name.startswith("wtemp") and var_name != "nao")
        ]
        # data = np.empty((len(var_names_part),
        #                  old_div(T_raw_data, self.sum_interval[0, 0])))
        # data[:] = np.nan
        sum_interval = self.sum_interval[0, 0]
        # mets = OrderedDict((name, self.met[name])
        #                    for name in var_names_part)

        def sum_to_nan(values):
            n_nans = np.isnan(values.astype(float)).sum()
            if n_nans > max_nans:
                return np.nan
            # return np.nanmean(values) * sum_interval
            return np.nansum(values)

        data_df = pd.DataFrame(
            data=self.met, index=self.times_orig, columns=var_names_part
        )
        if sum_interval > 1:
            data_df = data_df.resample("%dh" % sum_interval).agg(sum_to_nan)

        data = data_df.values.T
        times_ = data_df.index.to_pydatetime()

        if "U" in self.var_names and "U" not in self.met:
            # the wind has to be aggregated taking the direction into account
            data[self.var_names.index("U")] = daily_wind(
                self.met, self.sum_interval_dict["U"]
            )[1]

        return times_, data

    def _prepare_fixed_data(self):
        """Convert the data to standard-normal an put it into (K,T)-array form."""
        if self.fixed_variables:
            sh = shelve.open(self.seasonal_cache_file, "c")
            fixed_data = np.nan * np.empty((self.K, self.T_sim))
            for var_name, values in list(self.fixed_variables.items()):
                var_ii = self.var_names.index(var_name)
                if values is None:
                    # fix the input data the model was fitted on
                    values = self.data_raw[var_ii]
                solution_key = var_name

                # the fitting was originially done for daily sums of hourly
                # values, but we can have different aggregation lengths
                sum_interval = self.sum_interval_dict[var_name]
                if sum_interval != 24:
                    solution_key += "_%d" % sum_interval
                seas_class, dist_class, solution = sh[solution_key]
                dist = seas_class(
                    dist_class,
                    values,
                    self.times,
                    fixed_pars=conf.par_known[var_name],
                )
                quantiles = np.squeeze(dist.cdf(solution))
                transformed = distributions.norm.ppf(quantiles)
                fixed_data[var_ii] = transformed
            sh.close()
            return fixed_data
        else:
            return None

    def _gen_sim_times(
        self, T=None, start_str=None, stop_str=None, output_resolution=None
    ):
        """Generates an array of datetimes starting at start_date with T values
        and resolution hours in between.

        Parameters
        ----------
        T : int
            number of timesteps, optional
        resolution : int
            length of timestep in hours
        start_date : datetime object, optional
            Representation of first timestep
        output_resolution : int or float, optional
            Output resolution in hours

        Returns
        -------
        times_out : (T,) ndarray, dtype=object (datetime)

        Examples
        --------
        >>> import datetime
        >>> sim_times(4, 48, datetime.datetime(year=2000, month=1, day=1))
        array([2000-01-01 00:00:00, 2000-01-03 00:00:00, 2000-01-05 00:00:00,
              2000-01-07 00:00:00], dtype=object)
        """
        if T is None:
            T = self.T_sim
        if output_resolution is None:
            output_resolution = self.output_resolution
        # produce sim_doys and times_out:
        if start_str is None:
            if self.sim_times is None:
                time_first = self.times_orig[0]
                self.start_date = datetime.datetime(
                    time_first.year, time_first.month, time_first.day
                )
            else:
                self.start_date = self.sim_times[0]
        else:
            try:
                self.start_date = varwg.times.str2datetime(start_str)
            except ValueError:
                self.start_date = varwg.times.iso2datetime(start_str)
        if stop_str is not None:
            # overwrite T setting
            try:
                end_date = varwg.times.str2datetime(stop_str)
            except ValueError:
                end_date = varwg.times.iso2datetime(stop_str)
            t_diff_seconds = (end_date - self.start_date).total_seconds()
            T = int(t_diff_seconds / (60**2 * 24))
        # interval_secs = 60. ** 2 * output_resolution

        # times_out = np.cumsum([0] + (T - 1) * [interval_secs])
        # times_out += times.datetime2unix(self.start_date)
        # times_out = times.unix2datetime(times_out)
        resolution_timedelta = datetime.timedelta(
            hours=float(output_resolution)
        )
        times_out = np.array(
            [self.start_date + t * resolution_timedelta for t in range(T)]
        )
        return times_out

    def _fit_distribution(self, sh, var, var_name, solution_key, **kwds):
        try:
            seas_class = conf.seasonal_classes[var_name]
        except KeyError:
            raise RuntimeError(
                f"Configure seasonal_classes in {conf.__file__} for {var_name}."
            )
        if (
            issubclass(seas_class, skde.SeasonalKDE)
            or conf.dists[var_name] == "empirical"
        ):
            if self.verbose:
                print(f"\tFitting KDE to {var_name}")
            dist = seas_class(
                var,
                self.times,
                fixed_pars=conf.par_known[var_name],
                verbose=self.verbose,
                **kwds,
            )
            solution = dist.fit(
                # silverman=(var_name == "sun")
                silverman=(var_name in ("sun", "R"))
            )
            sh[solution_key] = [dist, None, solution]
        else:
            dist = seas_class(
                conf.dists[var_name],
                var,
                self.times,
                fixed_pars=conf.par_known[var_name],
                verbose=self.verbose,
                **kwds,
            )
            if self.verbose:
                print(f"\tFitting {dist} to {var_name}")
            solution = dist.fit()
            try:
                sh[solution_key] = [dist, conf.dists[var_name], solution]
            except TypeError:
                # we will rely on the distribution that is set
                # in the config file later
                sh[solution_key] = [dist, None, solution]
            if hasattr(dist, "supplements"):
                sh[solution_key + "suppl"] = dist.supplements
            if kwds.get("tabulate_cdf", False):
                cdf_table_key = solution_key + f"cdf_table_{dist.dist.name}"
                sh[cdf_table_key] = dist.cdf_table

        return dist, solution

    def _fit_seasonal(
        self, refit=None, values=None, doys=None, filter_nans=True
    ):
        if refit is None or refit is False:
            refit = tuple()
        elif refit == "all" or refit is True:
            refit = self.var_names
        sh = shelve.open(str(self.seasonal_cache_file), "c")
        try:
            keys = list(sh.keys())
        except Exception:
            print("Cache file corrupted, refitting...")
            os.remove(self.seasonal_cache_file)
            sh = shelve.open(str(self.seasonal_cache_file), "c")
            keys = []
        if values is None:
            values = self.data_raw
        if doys is None:
            doys = self.data_doys
        data_trans = np.empty_like(values)
        dist_sol = {}
        for var_ii, var in enumerate(values):
            var_name = self.var_names[var_ii]

            # py2/3 incompatibilities...
            solution_key = str(var_name)

            # the fitting was originially done for daily sums of
            # hourly values we can have different aggregation lengths
            # so we trigger a fitting if we come across one of these
            sum_interval = self.sum_interval_dict[var_name]
            plain_solution_key = solution_key
            if sum_interval != 24:
                solution_key += "_%d" % sum_interval

            kwds = (
                conf.dists_kwds[var_name]
                if var_name in conf.dists_kwds
                else {}
            )

            if solution_key not in keys or plain_solution_key in refit:
                # if self.verbose:
                #     print("\tFitting a distribution to ", var_name)
                dist, solution = self._fit_distribution(
                    sh, var, var_name, solution_key, **kwds
                )
            else:
                try:
                    dist, dist_class, solution = sh[solution_key]
                    if self.verbose:
                        print(
                            f"\tRecovered previous fit ({dist}) "
                            f"from shelve for: {var_name}"
                        )
                except (UnpicklingError, EOFError) as exc:
                    if self.verbose:
                        print(exc)
                    self._fit_distribution(
                        sh, var, var_name, solution_key, **kwds
                    )
                    dist, dist_class, solution = sh[solution_key]
                try:
                    supplements = sh[solution_key + "suppl"]
                except KeyError:
                    supplements = None
                    sh[solution_key + "suppl"] = None
                if kwds.get("tabulate_cdf", False):
                    if isinstance(dist_class, tuple):
                        dist_class = dist_class[1]
                    cdf_table_key = (
                        solution_key + f"cdf_table_{dist_class.name}"
                    )
                    try:
                        cdf_table = sh[cdf_table_key]
                    except KeyError:
                        cdf_table = None
                else:
                    cdf_table = None

                # var_ = var
                # if (
                #     issubclass(seas_class, skde.SeasonalKDE)
                #     or seas_class == "empirical"
                # ):
                #     dist = seas_class(
                #         var_,
                #         self.times,
                #         solution,
                #         fixed_pars=conf.par_known[var_name],
                #         **kwds,
                #     )
                # else:
                #     if dist_class is None:
                #         dist_class = conf.dists[var_name]
                #     dist = seas_class(
                #         dist_class,
                #         var_,
                #         self.times,
                #         solution=solution,
                #         fixed_pars=conf.par_known[var_name],
                #         supplements=supplements,
                #         cdf_table=cdf_table,
                #         **kwds,
                #     )

            # if self.verbose:
            #     print(
            #         "\tp-value of chi2 goodness-of-fit %.4f" % dist.chi2_test()
            #     )
            quantiles = dist.cdf(
                solution,
                x=var,
                doys=doys,
                # pdb=(pdb and (var_name != "R"))
            )
            assert len(quantiles) == len(var)
            data_trans[var_ii] = distributions.norm.ppf(quantiles)
            dist_sol[var_name] = dist, solution
        sh.close()

        if filter_nans:
            # we have outrageous outliers from time to time
            data_trans_finite = np.where(
                np.isfinite(data_trans), data_trans, 1e300
            )
            data_trans[np.abs(data_trans_finite) >= 1e300] = np.nan
            data_trans = my.interp_nonfin(data_trans, max_interp=3)
        return data_trans, dist_sol

    def _fit_seasonal_hourly(self, refit=None):
        if refit is None:
            refit = tuple()
        elif refit == "all" or refit is True:
            refit = self.var_names
        if isinstance(refit, str):
            refit = (refit,)

        # Fit hourly distributions to the data, if necesarry, and
        # qq-transform it to standard-norm.
        data_hourly_trans = []
        sh = shelve.open(str(self.seasonal_cache_file), "c")
        # sh.keys() is very slow
        fft_order = 20
        for var_name in self.var_names:
            # shelve has problems with unicode keys
            solution_key = str("%s_hourly" % var_name)
            values = self.met[var_name]
            dtimes = self.times_orig

            fixed_pars = conf.par_known_hourly[var_name]
            hour_neighbors = 12 if var_name == "R" else 4
            seas_class = conf.seasonal_classes_hourly[var_name]
            if solution_key not in sh or var_name in refit:
                if self.verbose:
                    print("\tFitting an hourly distribution to ", var_name)
                if seas_class in (skde.SeasonalKDE, skde.SeasonalHourlyKDE):
                    # fit hourly distributions
                    hourly_dist = skde.SeasonalHourlyKDE(
                        values,
                        dtimes,
                        fixed_pars=fixed_pars,
                        hour_neighbors=hour_neighbors,
                        verbose=self.verbose,
                    )
                    # for the time being, let's use scotts_rule of
                    # thumb and not the full blown leave_one_out
                    # maximum likelihood bandwidth estimation
                    solution = hourly_dist.fit(thumb=True)
                else:
                    hourly_dist = sd.SlidingDistHourly(
                        conf.dists_hourly[var_name],
                        values,
                        dtimes,
                        fixed_pars=fixed_pars,
                        verbose=self.verbose,
                        fft_order=fft_order,
                    )
                    solution = hourly_dist.fit()
                sh[solution_key] = solution
                sh.sync()
            else:
                if self.verbose:
                    print(
                        "\tRecover previous hourly fit from shelve for: "
                        + var_name
                    )
                solution = sh[solution_key]
                if seas_class in (skde.SeasonalKDE, skde.SeasonalHourlyKDE):
                    hourly_dist = skde.SeasonalHourlyKDE(
                        values,
                        dtimes,
                        solution=solution,
                        fixed_pars=fixed_pars,
                        hour_neighbors=hour_neighbors,
                    )
                else:
                    hourly_dist = sd.SlidingDistHourly(
                        conf.dists_hourly[var_name],
                        values,
                        dtimes,
                        solution=solution,
                        fixed_pars=fixed_pars,
                        fft_order=fft_order,
                    )
            qq = hourly_dist.cdf(solution, values, self.data_doys_raw)
            values_trans = distributions.norm.ppf(qq)
            data_hourly_trans += [values_trans]
            self.dist_sol[solution_key] = hourly_dist, solution
        sh.close()
        return data_hourly_trans

    def dist_sol_hourly(self, var_name):
        return self.dist_sol["%s_hourly" % var_name]

    def _wet_means_by_doy(
        self, non_rain_finite, rain_mask, doy_mask, fft_order
    ):
        wet_means_by_doy = np.empty((doy_mask.shape[0], len(non_rain_finite)))
        for doy_i, doy_mask_ in enumerate(doy_mask):
            wet_means_by_doy[doy_i] = np.nanmean(
                non_rain_finite[:, doy_mask_], axis=1
            )
        wet_means_by_doy = np.array(
            [
                my.fourier_approx(my.interp_nonfin(x), order=fft_order)
                for x in wet_means_by_doy.T
            ]
        ).T
        dry_doys = varwg.times.datetime2doy(self.times[~rain_mask])
        wet_means_by_doy = pd.DataFrame(
            wet_means_by_doy, index=np.arange(1, doy_i + 2)
        )
        return wet_means_by_doy.reindex(dry_doys).values.T

    def _wet_stds_by_doy(
        self, non_rain_finite, rain_mask, doy_mask, fft_order
    ):
        wet_stds_by_doy = np.empty((doy_mask.shape[0], len(non_rain_finite)))
        for doy_i, doy_mask_ in enumerate(doy_mask):
            wet_stds_by_doy[doy_i] = np.nanstd(
                non_rain_finite[:, doy_mask_], axis=1
            )
        wet_stds_by_doy = np.array(
            [
                my.fourier_approx(my.interp_nonfin(x), order=fft_order)
                for x in wet_stds_by_doy.T
            ]
        ).T
        dry_doys = varwg.times.datetime2doy(self.times[~rain_mask])
        wet_stds_by_doy = pd.DataFrame(
            wet_stds_by_doy, index=np.arange(1, doy_i + 2)
        )
        return wet_stds_by_doy.reindex(dry_doys).values.T

    def _betas_by_doy(self, X, y, rain_mask, doy_mask, fft_order):
        betas_by_doy = np.empty((doy_mask.shape[0], X.shape[1]))
        for doy_i, doy_mask_ in enumerate(doy_mask):
            X_doy = X[doy_mask_]
            y_doy = y[doy_mask_]
            beta = np.linalg.inv(X_doy.T @ X_doy) @ X_doy.T @ y_doy
            betas_by_doy[doy_i] = beta
        betas_by_doy = np.array(
            [
                my.fourier_approx(my.interp_nonfin(x), order=fft_order)
                for x in betas_by_doy.T
            ]
        ).T
        dry_doys = varwg.times.datetime2doy(self.times[~rain_mask])
        betas_by_doy = pd.DataFrame(
            betas_by_doy, index=np.arange(1, doy_i + 2)
        )
        return betas_by_doy.reindex(dry_doys).values.T

    def _negative_rain(
        self,
        data_trans,
        rain,
        doys,
        *,
        doy_width,
        fft_order,
        var_names=None,
        method="regression",
        kwds=None,
        self_name="R",
    ):
        """
        Transform rain-gaps to standard-normal by distance to wet conditions.

        Parameters
        ----------
        data_trans : 2d array
            Transformed variables.
        rain : 1d array
            Untransformed rain.
        doys : 1d array
        var_names : None or sequence of str, optional
            Non-rain variables to use.
        """
        if kwds is None:
            kwds = dict()
        if doys[1] - doys[0] < 1:
            rain_dist, sol = self.dist_sol_hourly(self_name)
        else:
            rain_dist, sol = self.dist_sol[self_name]

        if isinstance(rain_dist, distributions._Rain):
            rain_prob = rain_dist.all_parameters_dict(sol, doys)["rain_prob"]
        else:
            rain_prob = rain_dist.rain_probs(conf.threshold, doys)
        dry_prob = 1 - rain_prob

        rain_i = self.var_names.index(self_name)
        if var_names is None:
            var_names = [name for name in self.var_names if name != self_name]
        if ("abs_hum" in var_names) and ("abs_hum" not in self.var_names):
            if "rh" not in self.var_names and "theta" not in self.var_names:
                raise RuntimeError(
                    "Calculation of abs_hum in _negative_rain requires theta and rh"
                )
            abs_hum = meteox2y.rel2abs_hum(
                data_trans[self.var_names.index("rh")],
                data_trans[self.var_names.index("theta")],
            )
            var_names = tuple(
                var_name for var_name in var_names if var_name != "abs_hum"
            )
            non_rain = data_trans[
                [
                    self.var_names.index(var_name)
                    for var_name in var_names
                    if var_name != self_name
                ]
            ]
            non_rain = np.vstack((abs_hum[None, :], non_rain))
        else:
            non_rain = data_trans[
                [
                    self.var_names.index(var_name)
                    for var_name in var_names
                    if var_name != self_name
                ]
            ]
        threshold = conf.threshold
        rain_finite = np.where(np.isfinite(rain), rain, 0)
        rain_mask = rain_finite >= threshold
        dry_mask = ~rain_mask
        self.rain_mask = rain_mask
        doy_mask = sd.seasonal.build_doy_mask(doys, doy_width)

        def calc_dist_ranks_distance():
            # calculate means of non-rain variables during wet conditions
            non_rain_finite = np.where(np.isfinite(rain), non_rain, np.nan)
            wet_means = self._wet_means_by_doy(
                non_rain_finite, rain_mask, doy_mask, fft_order
            )
            # dry seasons are special!
            dry_means = self._wet_means_by_doy(
                non_rain_finite, dry_mask, doy_mask, fft_order
            )
            wet_means[:, rain_prob[dry_mask] < 0.05] = dry_means.mean(axis=1)[
                :, None
            ]
            wet_stds = self._wet_stds_by_doy(
                non_rain_finite, rain_mask, doy_mask, fft_order
            )

            # distance between current vector of non-rain variables
            # and mean of wet conditions
            distances = np.sum(
                (wet_means - non_rain[:, dry_mask]) / wet_stds, axis=0
            )
            # qq-transform the inverse distances to the lower tail of
            # the gaussian
            return my.rel_ranks(-distances)

        def calc_dist_ranks_regression():
            X = np.array(
                [
                    my.interp_nonfin(non_rain_var, max_interp=2)
                    for non_rain_var in non_rain
                ]
            ).T
            X[~np.isfinite(X)] = 0
            y = data_trans[rain_i]
            betas = self._betas_by_doy(X, y, rain_mask, doy_mask, fft_order)
            rain_reg = (non_rain[:, dry_mask] * betas).sum(axis=0)
            return my.rel_ranks(rain_reg)

        def calc_dist_ranks_simulation(p=3):
            rain_trans = np.where(rain_mask, data_trans[rain_i], np.nan)
            data_for_sim = np.vstack((non_rain, rain_trans))
            B, sigma_u = models.VAR_LS(data_for_sim, p)

            def bottom_stack(matrix):
                return np.moveaxis(np.squeeze(self.T_summed * [matrix]), 0, -1)

            Bs, sigma_us = map(bottom_stack, (B, sigma_u))
            try:
                A = np.linalg.cholesky(sigma_u)
            except np.linalg.LinAlgError:
                sigma_u.ravel()[:: self.K + 1] += (
                    varwg.rng.normal(self.K) * 1e-6
                )
                A = np.linalg.cholesky(sigma_u)
            varwg.reseed(0)
            data_infilled = models.SVAR_LS_fill(
                Bs, sigma_us, self.data_doys, data_for_sim, A=A
            )
            return my.rel_ranks(data_infilled[-1, dry_mask])

        match method:
            case "distance":
                dist_ranks = calc_dist_ranks_distance(**kwds)
            case "regression":
                dist_ranks = calc_dist_ranks_regression(**kwds)
            case "simulation":
                dist_ranks = calc_dist_ranks_simulation(**kwds)
            case _:
                raise RuntimeError(
                    "method must be one of: 'distance', 'regression', 'simulation'"
                )

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(nrows=2, ncols=1)
        # r_dist = calc_dist_ranks_distance()
        # r_regr = calc_dist_ranks_regression()
        # axs[0].plot(r_dist, label="dist")
        # axs[0].plot(r_regr, label="regr")
        # axs[1].scatter(r_dist, r_regr)
        # plt.show()

        self.rain_method = method
        self.neg_rain_doy_width = doy_width
        self.neg_rain_fft_order = fft_order

        # dryness probability in the standard-normal domain
        neg_rain = distributions.norm.ppf(dist_ranks * dry_prob[dry_mask])
        # normalize variance during dry spells
        # neg_rain *= (np.nanstd(data_trans[rain_i, rain_mask])
        #              / neg_rain.std())
        data_trans[rain_i, dry_mask] = neg_rain
        # data_trans[rain_i] /= np.nanstd(data_trans[rain_i])

        # assert np.all(data_old[rain_i, rain_mask] ==
        #               data_trans[rain_i, rain_mask])

        if self.plot:
            import matplotlib.pyplot as plt

            print("Wet means in std-n:")
            # calculate means of non-rain variables during wet conditions
            # non_rain = data_trans[non_rain_i]
            non_rain_finite = np.where(np.isfinite(rain), non_rain, np.nan)
            wet_means = self._wet_means_by_doy(
                non_rain_finite, rain_mask, doy_mask, fft_order
            )

            non_rain_var_names = (
                var_name for var_name in var_names if var_name != self_name
            )
            for var_name, wet_mean in zip(non_rain_var_names, wet_means):
                print("\t%s: %.3f" % (var_name, wet_mean.mean()))

            print(f"{self.var_names=}")
            print(f"{np.nanstd(data_trans[:, rain_mask], axis=1).round(3)=}")
            print(f"{np.nanstd(data_trans[:, dry_mask], axis=1).round(3)=}")
            print(f"{np.nanstd(data_trans, axis=1).round(3)=}")

            # fig, axs = plt.subplots(nrows=2)
            fig, ax = plt.subplots(nrows=1, ncols=1)
            axs = (ax,)
            axs[0].plot(
                self.times, data_trans[rain_i], "-x", label="rain trans"
            )
            axs[0].plot(self.times[dry_mask], neg_rain, "-x", label="neg rain")
            axs[0].plot(
                self.times,
                distributions.norm.ppf(1 - rain_prob),
                label="rain thresh",
            )
            # for var_i, var_name in enumerate(var_names):
            #     axs[0].plot(self.times[dry_mask],
            #                 wet_means[var_i], "-+",
            #                 label=f"wet_mean {var_name}")
            axs[0].plot(self.times, rain_prob, label="rain_prob")
            from varwg.time_series_analysis import phase_randomization as pr

            rain_sim = pr.randomize2d(
                data_trans,
                # taboo_period_min=150,
                # taboo_period_max=400
            )[rain_i]
            axs[0].plot(
                self.times, rain_sim, label="pr", linewidth=0.5, alpha=0.25
            )
            # dists_wet = np.sum(wet_means - non_rain[:, rain_mask], axis=0)
            # dists_ranks_wet = my.rel_ranks(-dists_wet)
            # neg_rain_wet = distributions.norm.ppf(dry_prob[rain_mask] +
            #                                       rain_prob[rain_mask] *
            #                                       dists_ranks_wet)
            # axs[0].plot(self.times[rain_mask], neg_rain_wet, "-x")
            # axs[0].plot(self.times, rain_reg)
            axs[0].legend(loc="best")
            axs[0].grid(True)

            # # axs[1].scatter(data_trans[rain_i, rain_mask],
            # #                neg_rain_wet, marker="x")
            # X = np.array([data_trans[i, rain_mask] for i in non_rain_i]).T
            # y = data_trans[rain_i, rain_mask].T
            # beta = np.linalg.inv(X.T @ X) @ X.T @ y
            # rain_reg = np.array(data_trans[non_rain_i].T * beta)
            # __import__('pdb').set_trace()
            # axs[1].scatter(
            #     data_trans[rain_i, rain_mask],
            #     rain_reg[rain_mask, rain_i],
            #     marker="+",
            #     facecolor="green",
            # )
            # axs[1].set_aspect("equal")

            # from varwg.time_series_analysis import time_series as ts
            # __import__('pdb').set_trace()
            # ts.plot_auto_corr(np.array((data_trans[rain_i], rain_sim)),
            #                   var_names=("obs", "sim"))
            # fig, ax = plt.subplots(nrows=1, ncols=1)
            # ax.acorr(data_trans[rain_i], usevlines=False, label="obs")
            # ax.acorr(rain_sim, usevlines=False, label="sim")
            # ax.legend(loc="best")

        return data_trans


if __name__ == "__main__":
    varwg.reseed(0)
    warnings.simplefilter("error", RuntimeWarning)
    import config_konstanz as conf
    import matplotlib.pyplot as plt

    import varwg

    varwg.conf = varwg.base.conf = varwg.core.plotting.conf = conf
    met_vg = varwg.VG(
        ("R", "theta", "ILWR", "Qsw", "rh", "u", "v"),
        # refit="R",
        verbose=True,
        neg_rain_doy_width=35,
        neg_rain_fft_order=3,
        plot=True,
    )
    met_vg.fit(p=3)
    simt, sim = met_vg.simulate()
    # met_vg.plot_exceedance_daily()
    plt.show()
