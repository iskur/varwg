"""Tools to fit seasonal distributions by describing the yearly cycle of
distribution parameters with triangular functions."""

from collections import abc
import hashlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats
import scipy.optimize as sp_optimize

from vg import helpers as my
from vg import smoothing, times
from vg.time_series_analysis import (
    distributions,
    optimize,
    seasonal,
    phase_randomization,
)
from tqdm import tqdm


class SeasonalDist(seasonal.Seasonal):
    def __init__(
        self,
        distribution,
        data,
        datetimes,
        fixed_pars=None,
        par_ntrig=None,
        time_form="%m",
        verbose=False,
        kill_leap=False,
        tabulate_cdf=False,
        **kwds,
    ):
        """distribution should be an object that implements pdf, cdf etc.

        fixed_pars is expected to be a dictionary mapping parameter
        names to functions that take days of the year as input to
        calculate distribution parameters.

        time_form (e.g. "%m") determines what is used to generate a starting
        solution of trigonometric parameters.

        var_ntrig should be a sequence mapping distribution parameters
        to number of trigonometric parameters. If that is None, 3 trig
        parameters per dist parameter are assumed

        """
        self.tabulate_cdf = tabulate_cdf
        super().__init__(data, datetimes, kill_leap=kill_leap)
        self.verbose = verbose
        if isinstance(distribution, abc.Iterable):
            self.dist = distribution[0](distribution[1], **kwds)
        else:
            self.dist = distribution
        self.fixed_pars = {} if fixed_pars is None else fixed_pars
        if not hasattr(self.dist, "parameter_names"):
            # we assume that we have come across a scipy.stats.distribution
            self._scipy_setup()

        if par_ntrig is None:
            self.par_ntrig = [3] * len(self.dist.parameter_names)
        else:
            self.par_ntrig = par_ntrig
        self.time_form = time_form

        self.non_fixed_names = [
            par_name
            for par_name in self.dist.parameter_names
            if par_name not in self.fixed_pars.keys()
        ]
        if not self.dist.isscipy and self.dist.supplements_names is not None:
            for suppl in self.dist.supplements_names:
                self.non_fixed_names.remove(suppl)

        self.n_trig_pars = 3
        self.pre_start = []
        self._supplements = self._params = self._medians_per_doy = None

    def __getstate__(self):
        dict_ = dict(self.__dict__)
        if "_scipy_setup" in dict_:
            del dict_["_scipy_setup"]
        return dict_

    def _scipy_setup(self):
        self.dist.isscipy = True
        self.dist.supplements_names = None
        self.dist.parameter_names = ["loc", "scale"]
        if self.dist.shapes is not None:
            self.dist.parameter_names = (
                self.dist.shapes.split(", ") + self.dist.parameter_names
            )
        self.dist.n_pars = len(self.dist.parameter_names)
        self.dist.__bases__ = distributions.Dist
        self.dist._fix_x = lambda x: x

        def _clean_kwds(kwds):
            return {
                key: value
                for key, value in list(kwds.items())
                if key in self.parameter_names
            }

        self.dist._clean_kwds = _clean_kwds

        def _constraints(x, **params):
            args = [params[name] for name in self.dist.parameter_names]
            if hasattr(self.dist, "_shape_info"):
                mask = np.full_like(args[0], False, dtype=bool)
                for shape_info in self.dist._shape_info():
                    param = params[shape_info.name]
                    lower, upper = shape_info.domain
                    if np.isclose(lower, 0):
                        lower = 1e-6
                    elif np.isneginf(lower):
                        lower = -1e3
                    mask[param < lower] = True
                    upper = min(upper, 1e3)
                    mask[param > upper] = True
                    # if np.any(mask):
                    #     __import__("pdb").set_trace()
                return mask
            return ~self.dist._argcheck(*args)

        self.dist._constraints = _constraints

    def trig2pars(self, trig_pars, _T=None):
        """This is the standard "give me distribution parameters for
        trigonometric parameters" that assumes 3 trig pars per distribution
        parameter."""
        if _T is None:
            try:
                _T = self._T
            except AttributeError:
                _T = self._T = (2 * np.pi / 365 * self.doys)[np.newaxis, :]
        # funny how the obvious thing looks so complicated
        a, b_0, phi = np.array(trig_pars).reshape(-1, 3).T[:, :, np.newaxis]
        # i lied, this took me some time. but now, the parameters are vertical
        # to T_ and we are broadcasting like some crazy shit pirate radio
        # station
        return a + b_0 * np.sin(_T + phi)

    def fixed_values_dict(self, doys=None):
        """This holds a cache for the self.doys values. If doys is given, those
        are used and the cache is left untouched."""

        def build_dict(doys):
            return dict(
                (fixed_par_name, func(doys))
                for fixed_par_name, func in self.fixed_pars.items()
            )

        if doys is None:
            try:
                return self._cached_fixed
            except AttributeError:
                self._cached_fixed = build_dict(self.doys)
            return self.fixed_values_dict()
        else:
            return build_dict(doys)

    def all_parameters_dict(self, trig_pars, doys=None):
        """Dictionary of the fixed, supplementary and fitted (or to be fitted)
        distribution paramters.

        """
        # copy (!) the fixed_values into params
        params = {
            key: val for key, val in list(self.fixed_values_dict(doys).items())
        }
        if doys is None:
            doys = self.doys
            _T = None
            x = self.data
        else:
            _T = np.copy(doys) * 2 * np.pi / 365
            x = np.full_like(_T, np.nan)

        # if isinstance(self, SlidingDist) and self._sliding_pars is not None:
        #     params_array = self._sliding_pars[self.doys2doys_ii(doys)].T
        # else:
        #     params_array = self.trig2pars(trig_pars, _T=_T)

        params_array = self.trig2pars(trig_pars, _T=_T)

        params.update(
            {
                param_name: values
                for param_name, values in zip(
                    self.non_fixed_names, params_array
                )
            }
        )

        if self.supplements is not None:
            params.update(
                {
                    param_name: values
                    for param_name, values in self.all_supplements_dict(
                        doys
                    ).items()
                }
            )

        # fft_approx can produce values outside of valid parameter
        # values.
        par_names_ordered = self.dist.parameter_names
        T = len(params[par_names_ordered[0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            invalid_mask = self.dist._constraints(x, **params)
            if np.all(invalid_mask):
                __import__("pdb").set_trace()
                self.dist._constraints(x, **params)

        # if np.all(invalid_mask):
        #     doys = np.atleast_1d(doys)
        #     # try to expand and provide the middle parameters
        #     doy_window = np.arange(1, self.doy_width + 1)
        #     doys_wide = np.concatenate((doys.min() - doy_window[::-1],
        #                                 doys,
        #                                 doys.max() + doy_window))
        #     doys_wide = doys_wide % (self.doys_unique[-1] + 1)
        #     return {name: np.squeeze(par[self.doy_width:-self.doy_width])
        #        for name, par
        #        in self.all_parameters_dict(trig_pars, doys_wide).items()}

        # fill up with raw, un-smoothed parameter values
        if np.any(invalid_mask):
            for name in self.non_fixed_names:
                if (
                    isinstance(self, SlidingDist)
                    and self._sliding_pars is not None
                ):
                    params[name][invalid_mask] = self._sliding_pars[
                        doys.astype(int) - 1
                    ][invalid_mask, self.non_fixed_names.index(name)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            invalid_mask = self.dist._constraints(x, **params)

        # interpolate over invalid values
        if np.any(invalid_mask):
            for name in self.non_fixed_names:
                par = params[name]
                par[invalid_mask] = np.nan
                half = T // 2
                par_pad = np.concatenate((par[-half:], par, par[:half]))
                params[name] = my.interp_nonfin(par_pad)[half:-half]
        # assert np.all(~self.dist._constraints(x, **params))
        return params

    def all_supplements_dict(self, doys=None):
        """Dictionary of all supplements assembled by doy distance."""
        if doys is None:
            doys = self.doys
        doys_ii = self.doys2doys_ii(doys)
        supplement_names = self.dist.supplements_names
        sup_dict = {}
        for name in supplement_names:
            value = [self.supplements[doy_ii][name] for doy_ii in doys_ii]
            # some supplements (e.g. kernel_data) contain lists of
            # differing lengths and cannot be converted to a single
            # array meaningfully. these are to remain as lists of
            # lists.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    # value = np.array(value)
                    value = np.asarray(value)
                except Warning:
                    pass
                except ValueError:
                    pass
            sup_dict[name] = value
        return sup_dict

    @property
    def monthly_grouped(self):
        try:
            return self._monthly_grouped
        except AttributeError:
            self._monthly_grouped = times.time_part_sort(
                self.datetimes, self.data, self.time_form
            )
        return self.monthly_grouped

    @property
    def monthly_grouped_x(self):
        try:
            return self._monthly_grouped[1]
        except AttributeError:
            # calling monthly grouped has the side effect of also calculating
            # monthly_grouped_x
            self.monthly_grouped
        return self.monthly_grouped[1]

    @property
    def daily_grouped_x(self):
        try:
            return self._daily_grouped_x
        except AttributeError:
            self._daily_grouped_x = times.time_part_sort(
                self.datetimes, self.data, "%j"
            )[1]
        return self.daily_grouped_x

    @property
    def monthly_params(self):
        """These are the distribution parameters per month. Returns a
        12xn_distribution_parameters array."""
        pars = []
        for month, sub_x in zip(*self.monthly_grouped):
            doys = self.doys[
                times.time_part(self.datetimes, self.time_form) == month
            ]
            fixed = {
                par_name: np.average(func(doys))
                for par_name, func in self.fixed_pars.items()
            }
            fitted_pars = self.dist.fit(sub_x, **fixed)

            if self.dist.supplements_names is not None:
                # remove supplements from pars_month
                fitted_pars = [
                    par
                    for name, par in zip(
                        self.dist.parameter_names, fitted_pars
                    )
                    if name not in self.dist.supplements_names
                ]
            pars += [fitted_pars]
        return np.array(pars)

    def get_start_params(self, opt_func=sp.optimize.fmin, x0=None, **kwds):
        """Estimate seasonal distribution parameters by fitting a
        trigonometric function to monthly lumped data. Lower and upper bounds
        (if they are defined in the distribution) will be fitted conservatively
        to at least ensure that the start parameters are a feasible solution.
        """
        if x0 is None:
            x0 = [0, 1, 0] + self.pre_start
        if "maxfun" not in kwds:
            kwds["maxfun"] = 1e5 * sum(self.par_ntrig)
        # TODO: friggin if shit
        if self.time_form == "%W":
            _T = np.linspace(0, 365, 54, endpoint=False)
        elif self.time_form == "%m":
            first_doms = times.str2datetime(
                ["01.%02d.2000 00:00:00" % month for month in range(1, 13)]
            )
            _T = np.array(times.time_part(first_doms, "%j"), dtype=float)
        _T *= 2 * np.pi / 365

        def error(trig_pars, emp_pars, times_=None):
            times_ = _T if times_ is None else times_
            return np.sum((self.trig2pars(trig_pars, times_) - emp_pars) ** 2)

        start_pars = []
        for par_name, par_timeseries in zip(
            self.dist.parameter_names, self.monthly_params.T
        ):
            # if the distribution comes with lower and upper bounds, find a
            # solution for those that contains all the measurement points.
            if par_name == "l" and "l" not in self.fixed_pars:
                minima = np.array([var.min() for var in self.daily_grouped_x])
                smoothed_mins = smoothing.smooth(minima, 365, periodic=True)
                # place the phase shift so that the minimum of the sine is at
                # the the minimum of the dataset
                mint = self.doys[np.argmin(smoothed_mins)]
                phi0 = 1.5 * np.pi - mint * 2 * np.pi / 365
                b_00 = 0.5 * (smoothed_mins.max() - smoothed_mins.min())
                a0 = smoothed_mins.mean()
                start = [a0, b_00, phi0] + self.pre_start
                lowest_diff = np.min(smoothed_mins - minima)
                k = 1.5
                while np.any(smoothed_mins < k * lowest_diff):
                    k += 0.5
                fit_me = smoothed_mins - k * lowest_diff
                times_ = np.arange(366.0) * 2 * np.pi / 365
                start_pars += list(
                    opt_func(
                        error, start, args=(fit_me, times_), disp=False, **kwds
                    )
                )
            elif par_name == "u" and "u" not in self.fixed_pars:
                maxima = np.array([var.max() for var in self.daily_grouped_x])
                smoothed_maxs = smoothing.smooth(maxima, 365, periodic=True)
                maxt = self.doys[np.argmax(smoothed_maxs)]
                phi0 = 0.5 * np.pi - maxt * 2 * np.pi / 365
                b_00 = 0.5 * (smoothed_maxs.max() - smoothed_maxs.min())
                a0 = smoothed_maxs.mean()
                start = [a0, b_00, phi0] + self.pre_start
                highest_diff = np.max(maxima - smoothed_maxs)
                fit_me = smoothed_maxs + 1.2 * highest_diff
                times_ = np.arange(366.0) * 2 * np.pi / 365
                start_pars += list(
                    opt_func(
                        error, start, disp=False, args=(fit_me, times_), **kwds
                    )
                )
            elif par_name not in self.fixed_pars:
                start_pars += list(
                    opt_func(
                        error, x0, args=(par_timeseries,), disp=False, **kwds
                    )
                )
        self._start_pars = start_pars
        return start_pars

    def _dist_method(
        self,
        func,
        trig_pars,
        data=None,
        doys=None,
        broadcast=False,
        delete_cache=False,
        build_table=False,
        **kwds,
    ):
        """Abstracts the common ground for self.{pdf,cdf,ppf} which call the
        according functions of the underlying distribution."""
        if (
            func.__name__ in ("cdf", "ppf")
            and self.tabulate_cdf
            and not build_table
        ):
            return self._dist_method_tabulated(func, data, doys)
        if delete_cache or build_table:
            self._params = None
        if self._params is not None:
            # check if it has the right length
            param_len = len(
                np.atleast_1d(self._params[list(self._params.keys())[0]])
            )
            if (data is not None) and (len(data) != param_len):
                self._params = None
            elif (doys is not None) and (len(doys) != param_len):
                self._params = None
            elif data is None and doys is None:
                self._params = None

        if self._params is None:
            params = self.all_parameters_dict(trig_pars, doys)
            params.update(kwds)
            self._params = params
        else:
            params = self._params
        data = np.atleast_1d(self.data if data is None else data)
        if broadcast:
            xx = data[np.newaxis, :]
            params = dict(
                (
                    (par_name, val)
                    if par_name == "kernel_data"
                    else (par_name, np.asarray(val)[:, np.newaxis])
                )
                for par_name, val in list(params.items())
            )
            if self.dist.isscipy:
                args = [
                    params[par_name] for par_name in self.dist.parameter_names
                ]
                return func(xx, *args)
            return func(xx, **params)
        else:
            if self.dist.isscipy:
                args = [
                    params[par_name] for par_name in self.dist.parameter_names
                ]
                # return np.vectorize(func)(data, *args)
                return func(data, *args)
            return func(data, **params)

    def _dist_method_tabulated(self, func, data=None, doys=None):
        match func.__name__:
            case "cdf":
                xx = self.cdf_table[..., 1]
                yy = self.cdf_table[..., 0]
            case "ppf":
                xx = self.cdf_table[..., 0]
                yy = self.cdf_table[..., 1]
            case _:
                RuntimeError(f"Expected a cdf or ppf function. Got {func}")
        if data is None:
            data = self.data
        if doys is None:
            doys = self.doys
        doys_ii = self.doys2doys_ii(doys)
        xx, yy = xx[doys_ii], yy[doys_ii]
        right_i = (data[:, None] < xx).argmax(axis=1)
        # for not wrapping around left_i
        right_i[right_i == 0] = 1
        left_i = right_i - 1
        ii = np.arange(xx.shape[0])
        x_left, x_right = xx[ii, left_i], xx[ii, right_i]
        y_left, y_right = yy[ii, left_i], yy[ii, right_i]
        interp = y_left + (data - x_left) * (y_right - y_left) / (
            x_right - x_left
        )
        interp[np.isnan(data)] = np.nan
        return interp

    def pdf(self, trig_pars, x=None, doys=None, **kwds):
        return self._dist_method(self.dist.pdf, trig_pars, x, doys, **kwds)

    def cdf(self, trig_pars, x=None, doys=None, **kwds):
        return self._dist_method(self.dist.cdf, trig_pars, x, doys, **kwds)

    def ppf(
        self, trig_pars, quantiles=None, doys=None, mean_shift=None, **kwds
    ):
        if mean_shift is None:
            return self._dist_method(
                self.dist.ppf, trig_pars, quantiles, doys, **kwds
            )
        else:
            # normalize quantiles and shift the target distribution
            # normalize quantiles
            stdn = distributions.norm.ppf(quantiles)
            bias = distributions.norm.ppf(self.cdf(trig_pars).mean())
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                quantiles = distributions.norm.cdf(stdn - stdn.mean() + bias)
            # TODO: neither "loc" nor "mu" have to be the mean of the
            # distribution directly! This needs to be more
            # sophisticated for a lot of distributions.
            # mean_par_name = "loc" if self.dist.isscipy else "mu"
            # mean_par_i = self.dist.parameter_names.index(mean_par_name)
            # trig_pars = np.copy(trig_pars)
            # trig_pars[mean_par_i, 0] += mean_shift * 2 * trig_pars.shape[1]
            # xx = self._dist_method(
            #     self.dist.ppf, trig_pars, quantiles, doys, **kwds
            # )
            # if mean_shift > 0:
            #     __import__("pdb").set_trace()
            return (
                self._dist_method(
                    self.dist.ppf, trig_pars, quantiles, doys, **kwds
                )
                + mean_shift
            )

    def mean(self, trig_pars, doys=None, **kwds):
        params = self.all_parameters_dict(trig_pars, doys)
        params.update(kwds)
        parameter_names = list(params.keys())
        parameter_values = list(params.values())
        means = np.empty(len(parameter_values[0]), dtype=float)
        for doy_i, pars_doy in enumerate(zip(*parameter_values)):
            par_dict_doy = {
                name: value for name, value in zip(parameter_names, pars_doy)
            }
            means[doy_i] = self.dist.mean(**par_dict_doy)
        return means

    def median(self, trig_pars, doys=None, **kwds):
        if self._medians_per_doy is None:
            if self.tabulate_cdf:
                medians_per_doy = self.ppf(
                    trig_pars,
                    quantiles=np.full(len(self.doys_unique), 0.5),
                    doys=self.doys_unique,
                )
            else:
                medians_per_doy = np.empty(len(self.doys_unique), dtype=float)
                params = self.all_parameters_dict(trig_pars, self.doys_unique)
                params.update(kwds)
                parameter_names = list(params.keys())
                parameter_values = list(params.values())
                for doy_i, pars_doy in enumerate(zip(*parameter_values)):
                    par_dict_doy = {
                        name: value
                        for name, value in zip(parameter_names, pars_doy)
                    }
                    medians_per_doy[doy_i] = self.dist.median(**par_dict_doy)
            self._medians_per_doy = medians_per_doy
        if doys is None:
            doys = self.doys
        return self._medians_per_doy[self.doys2doys_ii(doys)]

    # def qq_shift(self, theta_incr, trig_pars, x=None, doys=None, **kwds):
    #     """Empirical estimation of shift in std-normal for given theta_incr."""
    #     # executing this method is expensive and in the context of
    #     # weathercop's conditional simulation, might happen often with
    #     # the same theta_incr, so do some caching here to return known
    #     # results
    #     # theta_incr_key = hash(round(theta_incr[0], 6))
    #     # theta_incr_key = hashlib.md5(str(round(theta_incr[0], 6)).encode()).hexdigest()
    #     qq = self.cdf(trig_pars, x=x, doys=doys, **kwds)
    #     zero = 1e-6
    #     one = 1 - zero
    #     stdn = distributions.norm.ppf(np.minimum(np.maximum(qq, zero), one))
    #     stdn = phase_randomization.randomize2d(np.array([stdn])).squeeze()
    #     data = np.atleast_1d(self.data if x is None else x)
    #     data_mean = data.mean()

    #     def c2incr(c):
    #         xx_act = self.ppf(
    #             trig_pars,
    #             quantiles=distributions.norm.cdf(stdn + c),
    #             doys=doys,
    #             kwds=kwds,
    #         )
    #         return (xx_act.mean() - data_mean - theta_incr) ** 2

    #     if self.verbose:
    #         print(f"\tFilling shift-cache for {theta_incr=}")
    #     result = sp_optimize.minimize_scalar(c2incr)

    #     # fig, ax = plt.subplots(nrows=1, ncols=1)
    #     # xx_act = self.ppf(trig_pars,
    #     #                   quantiles=distributions.norm.cdf(stdn + result.x),
    #     #                   doys=doys,
    #     #                   kwds=kwds)
    #     # ax.axvline(data_mean, label="data_mean", color="k")
    #     # ax.axvline(xx_act.mean(), label="xx_act", color="b")
    #     # ax.axvline(data_mean + theta_incr, label="data_mean + theta_incr",
    #     #            color="k", linestyle="--")
    #     # ax.legend()
    #     # plt.show()

    #     return result.x

    def _clear_qq_shift_cache(self):
        self._qq_shift_cache = None

    @property
    def solution(self):
        try:
            return self._solution
        except AttributeError:
            # self.fit sets the self._solution attribute
            self.fit()
            return self.solution

    @solution.setter
    def solution(self, sol):
        self._solution = sol

    @property
    def start_pars(self):
        try:
            return self._start_pars
        except AttributeError:
            self._start_pars = self.get_start_params()
        return self.start_pars

    def fit(
        self, data=None, opt_func=optimize.simulated_annealing, x0=None, **kwds
    ):
        if data is not None:
            self.data = data
        if x0 is None:
            x0 = self.get_start_params(maxfun=1e3 * sum(self.par_ntrig))

        def constraints(trig_pars):
            dist_params = self.all_parameters_dict(trig_pars)
            return self.dist._constraints(self.data, **dist_params)

        def unlikelihood(trig_pars):
            densities = self.pdf(trig_pars) + 1e-12
            obj_value = -np.sum(np.log(densities))
            if not np.isfinite(obj_value):
                raise ValueError("Non-finite objective function value.")
            return obj_value

        def chi2(trig_pars):
            quantiles = self.cdf(trig_pars, delete_cache=True)
            f_obs = np.histogram(quantiles, 40)[0].astype(float)
            f_obs /= f_obs.sum()
            f_exp = np.array([float(len(f_obs))] * len(f_obs))
            obj_value = sp.stats.chisquare(f_obs, f_exp)[0]
            if not np.isfinite(obj_value):
                raise ValueError
            return obj_value

        #        n = len(self.data)
        #        x_sorted_ii = np.argsort(self.data)
        #        ranks_plus = np.arange(0., n) / n
        #        ranks_minus = np.arange(1., n + 1) / n
        #        def ks(trig_pars):
        #            cdf_values = self.cdf(trig_pars)[x_sorted_ii]
        #            #cdf_values[np.isnan(cdf_values)] = np.inf
        #            dmin_plus = np.abs(cdf_values - ranks_plus).max()
        #            dmin_minus = np.abs(cdf_values - ranks_minus).max()
        #            return max(dmin_plus, dmin_minus)
        #        def f_diff(trig_pars):
        #            cdf_values = self.cdf(trig_pars)[x_sorted_ii]
        #            #cdf_values[np.isnan(cdf_values)] = np.inf
        #            dmin_plus = np.sum((cdf_values - ranks_plus) ** 2)
        #            dmin_minus = np.sum((cdf_values - ranks_minus) ** 2)
        #            return dmin_plus + dmin_minus
        #        def combined(trig_pars):
        #            return unlikelihood(trig_pars) + f_diff(trig_pars)
        result = opt_func(
            unlikelihood,
            x0,
            constraints=(constraints,),
            callback=lambda data: setattr(self, "_solution", data),
            **kwds,
        )
        if opt_func is sp.optimize.anneal:
            print("retval was %d" % result[-1])
            result = result[0]
        self._solution = result
        return result

    def chi2_test(self, k=None):
        """Chi-square goodness-of-fit test.
        H0: The given data **data** follows **distribution** with parameters
            aquired by ``func::SeasonalDist.fit``
        To side-step complications arising from having a different
        distribution for every doy, we test whether the quantiles (which are
        deseasonalized) are evenly distributed.

        Parameters
        ----------
        k : int
            Number of classes (bins)

        Returns
        -------
        p_value : float
        """
        quantiles = self.cdf(
            self.solution, x=self.data, doys=self.doys, build_table=True
        )
        n = len(quantiles)
        n_parameters = len(self.dist.parameter_names)
        if k is None:
            # k = int(n ** .5)
            k = n_parameters + 2
        observed = np.histogram(quantiles[np.isfinite(quantiles)], k)[0]
        expected = float(n) / k
        chi_test = np.sum((observed - expected) ** 2 / expected)
        # degrees of freedom:
        dof = k - n_parameters - 1
        return stats.chi2.sf(chi_test, dof)

    def scatter_cdf(
        self, trig_pars=None, figsize=None, title=None, *args, **kwds
    ):
        if trig_pars is None:
            try:
                trig_pars = self._solution
            except AttributeError:
                trig_pars = self.start_pars
        doys = np.arange(1, 366, dtype=float)
        _T = doys * 2 * np.pi / 365
        xx = np.linspace(self.data.min(), self.data.max(), 100)
        quants = self.cdf(trig_pars, xx, doys, broadcast=True)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.gca()
        co = ax1.contourf(doys, xx, quants.T, 15)
        plt.scatter(
            self.doys,
            self.data,
            marker="o",
            facecolors=(0, 0, 0, 0),
            edgecolors=(0, 0, 0, 0.5),
        )
        # plot lower and upper bounds
        for fixed_values in list(self.fixed_values_dict(doys).values()):
            plt.plot(doys, fixed_values, "r")
        try:
            for par in (
                trig_pars[2 * self.n_trig_pars : 3 * self.n_trig_pars],
                trig_pars[3 * self.n_trig_pars : 4 * self.n_trig_pars],
            ):
                plt.plot(doys, np.squeeze(self.trig2pars(par, _T)), "r")
        except (IndexError, ValueError):
            pass
        plt.colorbar(co)
        plt.xlim(0, len(doys))
        plt.ylim(xx.min(), xx.max())
        plt.xlabel("Day of Year")
        plt.grid()
        plt.legend()
        if title is not None:
            plt.title(title)
        return fig

    def scatter_pdf(
        self,
        trig_pars=None,
        figsize=None,
        title=None,
        opacity=0.25,
        s_kwds=None,
        n_sumup=24,
        *args,
        **kwds,
    ):
        plt.set_cmap("coolwarm")
        if trig_pars is None:
            try:
                trig_pars = self._solution
            except AttributeError:
                trig_pars = self.start_pars
        if s_kwds is None:
            s_kwds = dict(marker="o")
        _T = self.doys_unique * 2 * np.pi / 365
        xx = np.linspace(self.data.min(), self.data.max(), 100)
        dens = self.pdf(trig_pars, xx, self.doys_unique, broadcast=True)
        xx /= n_sumup
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax1.contourf(self.doys_unique, xx, dens.T, 15)
        ax1.scatter(
            self.doys,
            self.data / n_sumup,
            facecolors=(0, 0, 0, 0),
            edgecolors=(0, 0, 0, opacity),
            **s_kwds,
        )

        # plot lower and upper bounds
        for fixed_name, fixed_values in self.fixed_values_dict(
            self.doys_unique
        ).items():
            if fixed_name == "kernel_bounds":
                continue
            plt.plot(self.doys_unique, fixed_values / n_sumup, "r")
        try:
            for par in (
                trig_pars[2 * self.n_trig_pars : 3 * self.n_trig_pars],
                trig_pars[3 * self.n_trig_pars : 4 * self.n_trig_pars],
            ):
                plt.plot(
                    self.doys_unique,
                    np.squeeze(self.trig2pars(par, _T)) / n_sumup,
                    "r",
                )
        except (IndexError, ValueError):
            pass

        ax1.set_xlim(0, 366)
        ax1.set_ylim(xx.min(), xx.max())
        self._set_monthly_ticks(ax1)
        ax1.grid()

        # plot additional parameters for RainMix
        if isinstance(self.dist, distributions._Rain):
            ax2 = ax1.twinx()
            pars = self.all_parameters_dict(trig_pars, self.doys_unique)
            for par_name in ("rain_prob", "q_thresh"):
                if par_name in pars:
                    ax2.plot(self.doys_unique, pars[par_name], label=par_name)
            ax2.set_xlim(0, 366)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("probabilities")
            ax2.legend(loc="best")
            if "f_thresh" in pars:
                ax1.plot(self.doys_unique, pars["f_thresh"], label="f_thresh")

        if title is not None:
            fig.suptitle(title)
        return fig, ax1

    def plot_seasonality_fit(self, solution=None):
        if self.time_form == "%m":
            doys = np.array(
                times.time_part(
                    times.str2datetime(
                        ["2011 %d" % month for month in range(1, 13)], "%Y %m"
                    ),
                    "%j",
                ),
                dtype=float,
            )
        elif self.time_form == "%W":
            doys = times.datetime2doy(
                times.str2datetime(
                    ["2011 %d" % week for week in range(0, 54)], "%Y %W"
                )
            )

        _T = 2 * np.pi * doys / 365
        if solution is None:
            solution = self._solution
        dist_params = self.trig2pars(solution, _T)
        figs, axs = [], []
        for par_name, emp_pars, fit_pars in zip(
            self.dist.parameter_names, self.monthly_params.T, dist_params
        ):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(emp_pars, label="empirical")
            ax.plot(fit_pars, label="fitted")
            ax.set_title(par_name)
            ax.legend()
            figs += [fig]
            axs += [ax]
        return figs, axs

    def plot_monthly_fit(
        self, solution=None, n_classes=30, figsize=(23, 12), dists_alt=None
    ):
        """
        Parameters
        ----------
        solution: None, "monthly_fitted" or ndarray
            If "monthly_fitted" don't use the fft approximation of
            seasonality, but monthly fit to the data.
        """
        # get the doys of the middle of the months very roughly
        month_doys = np.linspace(1, 366, 12, endpoint=False).astype(int) + 15
        if solution == "monthly_fitted":
            monthly_params = self.monthly_params
        else:
            if solution is None:
                solution = self.solution
            # get a parameter set per month from the supplied solution
            monthly_params_dict = self.all_parameters_dict(
                solution, month_doys
            )
            par_names = self.dist.parameter_names
            if self.dist.supplements_names is not None:
                par_names = [
                    name
                    for name in par_names
                    if name not in self.dist.supplements_names
                ]
            monthly_params = my.list_transpose(
                [monthly_params_dict[par_name] for par_name in par_names]
            )
            monthly_params = np.array(monthly_params)
        monthly_fixed = [
            {
                par_name: func(month_doy)
                for par_name, func in self.fixed_pars.items()
            }
            for month_doy in month_doys
        ]

        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=figsize, sharex=True)
        axs = axs.ravel()
        fig.suptitle(self.dist.name)
        try:
            fig.canvas.set_window_title(self.dist.name)
        except AttributeError:
            pass
        for ii, values in enumerate(self.monthly_grouped_x):
            ax1 = axs[ii]

            # the histogram of the data
            bins = ax1.hist(
                values, n_classes, density=True, facecolor="grey", alpha=0.75
            )[1]

            class_middles = 0.5 * (bins[1:] + bins[:-1])
            if self.dist.isscipy:
                density = self.dist.pdf(class_middles, *monthly_params[ii])
                monthly_dict = dict()
            else:
                parameter_names = self.dist.parameter_names
                if self.dist.supplements_names is not None:
                    parameter_names = [
                        name
                        for name in parameter_names
                        if name not in self.dist.supplements_names
                    ]
                monthly_dict = dict(
                    (par_name, pars)
                    for par_name, pars in zip(
                        parameter_names, monthly_params[ii]
                    )
                )
                monthly_dict.update(monthly_fixed[ii])
                if isinstance(self.dist, distributions.RainMix):
                    month_i = month_doys[ii]
                    for key in (
                        "q_thresh",
                        "q_kde_eval",
                        "x_eval",
                        "f_thresh",
                    ):
                        monthly_dict[key] = self.supplements[month_i][key]
                if self.dist.supplements_names is not None:
                    f_thresh = monthly_dict["f_thresh"]
                    monthly_dict["kernel_data"] = values[values >= f_thresh]
                density = self.dist.pdf(class_middles, **monthly_dict)
            ax1.plot(class_middles, density, "r--")

            # the quantile part
            ax2 = ax1.twinx()
            # empirical cdf
            values_sort = np.sort(values)
            ranks_emp = (0.5 + np.arange(len(values))) / len(values)
            ax2.plot(values_sort, ranks_emp)
            # theoretical cdf
            ranks_gen = np.linspace(1e-6, 1 - 1e-6, 100)
            xx = ranks_gen
            if self.dist.isscipy:
                ranks_theory = self.dist.cdf(xx, *monthly_params[ii])
                xx_from_ppf = self.dist.ppf(ranks_gen, *monthly_params[ii])
                qq = self.dist.cdf(values, *monthly_params[ii])
            else:
                ranks_theory = self.dist.cdf(xx, **monthly_dict)
                xx_from_ppf = self.dist.ppf(ranks_gen, **monthly_dict)
                qq = self.dist.cdf(values, **monthly_dict)
            ax2.plot(xx, ranks_theory, "r--")
            ax2.plot(xx_from_ppf, ranks_gen, "g--")
            ax2.hist(
                qq[np.isfinite(qq)],
                40,
                color="gray",
                density=True,
                histtype="step",
                orientation="horizontal",
            )
            if "f_thresh" in monthly_dict:
                ax2.axvline(
                    f_thresh, linestyle="--", linewidth=1, color="gray"
                )
            if hasattr(self.dist, "q_thresh"):
                ax2.axhline(
                    self.dist.q_thresh,
                    linestyle="--",
                    linewidth=1,
                    color="gray",
                )

            if dists_alt:
                if not isinstance(dists_alt, abc.Iterable):
                    dists_alt = (dists_alt,)

                for dist_alt in dists_alt:
                    dist = dist_alt(*dist_alt.fit(values))
                    ax1.plot(class_middles, dist.pdf(class_middles), "--")
                    ax2.plot(xx, dist.cdf(xx), "--")

            params_str = ", ".join(
                (
                    f" {par_name}: None"
                    if par is None
                    else f" {par_name}: {par:.3f}"
                )
                for par_name, par in zip(
                    self.dist.parameter_names, monthly_params[ii]
                )
                if (
                    self.dist.supplements_names is not None
                    and par_name not in self.dist.supplements_names
                )
            )
            ax1.set_title("month:%d %s" % (ii + 1, params_str), fontsize=11)
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
            ax2.set_ylim(0, 1)
        fig.tight_layout()
        return fig, axs

    # def plot_monthly_params(self):
    #     n_pars = self.dist.n_pars - n_fixed_pars
    #     fig, axs = plt.subplots(nrows=n_pars, ncols=1)
    #     for par_name, values, ax in zip(self.dist.parameter_names,
    #                                  self.monthly_params.T,
    #                                  axs):
    #         ax.plot(values)
    #         ax.title(par_name)
    #     return fix, axs


class SlidingDist(SeasonalDist):
    """Can we get a better picture at the seasonalities of the distribution
    parameters if we estimate them over a sliding window over the doys?"""

    def __init__(
        self,
        distribution,
        x,
        dtimes,
        doy_width=15,
        fft_order=4,
        solution=None,
        cdf_table=None,
        supplements=None,
        *args,
        **kwds,
    ):
        super().__init__(distribution, x, dtimes, *args, **kwds)
        self.doy_width, self.fft_order = doy_width, fft_order
        self.cdf_table = cdf_table
        # usefull to assess goodness-of-fit/overfitting
        self.n_trig_pars = 2 * fft_order

        self.solution = solution
        self.supplements = supplements
        self._doy_mask = self._sliding_pars = None
        self._qq_shift_cache = None

    def __str__(self):
        return f"SlidingDist({self.dist.name})"

    @property
    def doy_mask(self):
        """Returns a (n_unique_doys, len(data)) ndarray"""
        if self._doy_mask is None:
            self._doy_mask = seasonal.build_doy_mask(
                self.doys, self.doy_width, self.doys_unique
            )
        return self._doy_mask

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

    @property
    def sliding_pars(self):
        if self._sliding_pars is None:
            n_fixed_pars = len(
                list(self.dist._clean_kwds(self.fixed_pars).keys())
            )
            n_pars = self.dist.n_pars - n_fixed_pars
            self._sliding_pars = np.ones((self.n_doys, n_pars))
            if self.dist.supplements_names:
                # we need to save supplements also (needed for method
                # calling, but not fitted)
                self._supplements = []
            for doy_i in tqdm(range(self.n_doys), disable=(not self.verbose)):
                data = self.data[self.doy_mask[doy_i]]
                if not self.dist.isscipy:
                    # # weight the data by doy-distance
                    # doys = self.doys[self.doy_mask[doy_i]]
                    # doy_dist = times.doy_distance(doy_i + 1, doys)
                    # weights = (1 - doy_dist /
                    #            (self.doy_width + 2)) ** 2
                    weights = np.ones_like(data)
                fixed = {
                    par_name: func(self.doys_unique[doy_i])
                    for par_name, func in list(self.fixed_pars.items())
                }
                if self.dist.isscipy or isinstance(
                    self.dist, distributions.Rain
                ):
                    if doy_i > 0:
                        loc = self._sliding_pars[doy_i - 1, -2]
                        scale = self._sliding_pars[doy_i - 1, -1]
                        initial = self._sliding_pars[doy_i - 1, :-2]
                        try:
                            sol = self.dist.fit(
                                data, *initial, loc=loc, scale=scale
                            )
                        except stats.FitError:
                            sol = self.dist.fit(data)
                    else:
                        sol = self.dist.fit(data, **fixed)
                    sol_dict = {
                        name: value
                        for name, value in zip(self.dist.parameter_names, sol)
                    }
                    if self.dist._constraints(
                        data,
                        **sol_dict,
                    ):
                        sol = np.full_like(sol, np.nan)
                    self._sliding_pars[doy_i] = sol
                else:
                    if doy_i > 0:
                        x0 = self._sliding_pars[doy_i - 1]
                    else:
                        x0 = [np.nan] * n_pars
                    if np.any(np.isnan(x0)):
                        x0 = self.dist.fit(data, **fixed)
                    result = self.dist.fit_ml(
                        data, weights=weights, x0=x0, method="Powell", **fixed
                    )
                    if (
                        None in result.x
                        or np.nan in result.x
                        or not result.success
                    ):
                        sol = [np.nan] * n_pars
                    else:
                        sol = result.x
                    try:
                        self._sliding_pars[doy_i] = sol
                    except ValueError:
                        print(
                            "Could not fit seasonal distribution. "
                            "Check your conf-file!"
                        )
                        raise

                    if result.supplements:
                        self._supplements += [result.supplements]
            if self.verbose:
                print()

        # try to interpolate over bad fittings
        pars = self._sliding_pars
        if isinstance(self.dist, stats._continuous_distns.exponnorm_gen):
            # there are sometimes K values that cause problems
            pars[pars[:, 0] < 1e-6, 0] = 1e-6
        # pars[pars > 1e9] = np.nan
        nan_cols = np.where(np.isnan(pars).sum(axis=1) > 0)[0]
        # i don't trust the nan-adjacent!
        if len(nan_cols):
            # left_cols = np.maximum(nan_cols - self.doy_width // 4, 0)
            # right_cols = np.minimum(nan_cols + self.doy_width // 4,
            #                         pars.shape[0] - 1)
            left_cols = np.maximum(nan_cols - 2, 0)
            right_cols = np.minimum(nan_cols + 2, pars.shape[0] - 1)
            pars[left_cols] = np.nan
            pars[right_cols] = np.nan
        for par_i, par in enumerate(pars.T):
            if np.any(np.isnan(par)):
                half = len(par) // 2
                par_pad = np.concatenate((par[-half:], par, par[:half]))
                interp = my.interp_nonfin(par_pad)[half:-half]
                self._sliding_pars[:, par_i] = interp

        if self.dist.supplements_names is None:
            return self._sliding_pars.T

        # do the same for supplements
        for sup_i, sup_name in enumerate(self.dist.supplements_names):
            values = [
                self._supplements[doy_i][sup_name]
                for doy_i in range(self.n_doys)
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    values = np.array(values, dtype=float)
                except Warning:
                    continue
                except ValueError:
                    continue
            if values.ndim > 1:
                continue
            if np.any(np.isnan(values)):
                half = len(values) // 2
                values_pad = np.concatenate(
                    (values[-half:], values, values[:half])
                )
                interp = my.interp_nonfin(values_pad)[half:-half]
                for day_ii in range(self.n_doys):
                    self._supplements[day_ii][sup_name] = interp[day_ii]
        return self._sliding_pars.T

    @property
    def supplements(self):
        if self._supplements is None:
            # this causes supplements to be assembled as a side-effect
            self.solution
        return self._supplements

    @supplements.setter
    def supplements(self, suppl):
        self._supplements = suppl

    @property
    def solution(self):
        if self._solution is None:
            trans = [np.fft.rfft(sl_par) for sl_par in self.sliding_pars]
            self._solution = np.array(trans)
        return self._solution

    @solution.setter
    def solution(self, trans):
        self._solution = trans

    @solution.deleter
    def solution(self):
        self._solution = None
        self._sliding_pars = None

    def fourier_approx(self, fft_order=4, trig_pars=None):
        if trig_pars is None:
            trig_pars = self.solution

        _fourier_approx = np.empty(
            (len(self.dist.parameter_names), self.n_doys)
        )
        for ii, trans_par in enumerate(trig_pars):
            _fourier_approx[ii] = np.fft.irfft(
                trans_par[: fft_order + 1], self.n_doys
            )
        # enforce domain restrictions, which can be violated because
        # of fourier approximation
        if (bounds := getattr(self.dist, "_bounds", None)) is not None:
            for par_i, (par, (lower, upper)) in enumerate(
                zip(_fourier_approx, bounds)
            ):
                par[par <= lower] = lower
                par[par >= upper] = upper
                _fourier_approx[par_i] = par
        elif hasattr(self.dist, "_shape_info"):
            for par_i, (shape_info, par) in enumerate(
                zip(self.dist._shape_info(), _fourier_approx)
            ):
                lower, upper = shape_info.domain
                # unfortunately, being within the domain does not
                # necessarily mean that numpy can handle it
                if np.isclose(lower, 0):
                    lower = 1e-6
                elif np.isneginf(lower):
                    lower = -1e6
                par[par <= lower] = lower
                upper = min(upper, 1e6)
                par[par >= upper] = upper
                _fourier_approx[par_i] = par
        return _fourier_approx

    def fit(self, x=None, **kwds):
        if x is not None:
            self.data = x
        solution = self.solution
        if self.tabulate_cdf and self.cdf_table is None:
            self.cdf_table = self._tabulate_cdf(solution)
        return solution

    def _tabulate_cdf(self, solution, n_points=1000):
        table = np.full((len(self.doys_unique), n_points, 2), np.nan)
        data_min, data_max = self.data.min(), self.data.max()
        extra = 0.05 * (data_max - data_min)
        xx = np.linspace(data_min - extra, data_max + extra, n_points)
        if self.verbose:
            print("Building cdf table")
        for doy_i, doy in tqdm(
            enumerate(self.doys_unique),
            disable=not self.verbose,
            total=len(self.doys_unique),
        ):
            qq = self.cdf(
                solution, xx, doys=np.full(n_points, doy), build_table=True
            )
            table[doy_i, :, 0] = qq
            table[doy_i, :, 1] = self.dist._fix_x(xx)
        return table

    def trig2pars(self, trig_pars, _T=None, fft_order=None):
        fft_order = self.fft_order if fft_order is None else fft_order
        if _T is None:
            try:
                _T = self._T
            except AttributeError:
                _T = self._T = (2 * np.pi / 365 * self.doys)[np.newaxis, :]
        doys = np.atleast_1d(365 * np.squeeze(_T) / (2 * np.pi))
        doys_ii = self.doys2doys_ii(doys)
        fourier_pars = self.fourier_approx(fft_order, trig_pars)
        return np.array([fourier_pars[:, doy_i] for doy_i in doys_ii]).T

    def plot_fourier_fit(self, fft_order=None):
        """Plots the Fourier approximation of all parameters."""

        month_doys = np.linspace(1, 366, 12, endpoint=False).astype(int) + 15
        # get a parameter set per month
        monthly_params_dict = self.all_parameters_dict(
            self.solution, month_doys
        )
        par_names = self.dist.parameter_names
        if self.dist.supplements_names is not None:
            par_names = [
                name
                for name in par_names
                if name not in self.dist.supplements_names
            ]
        monthly_params = my.list_transpose(
            [monthly_params_dict[par_name] for par_name in par_names]
        )
        monthly_params = np.array(monthly_params)

        fft_order = self.fft_order if fft_order is None else fft_order
        fig, axs = plt.subplots(
            len(self.non_fixed_names), sharex=True, squeeze=True
        )
        pars = self.fourier_approx(fft_order)
        if pars.shape[1] > 1:
            for par_i, par_name in enumerate(self.non_fixed_names):
                axs[par_i].plot(self.doys_unique, self.sliding_pars[par_i])
                axs[par_i].plot(self.doys_unique, pars[par_i])
                axs[par_i].grid(True)
                axs[par_i].set_title(
                    "%s Fourier fft_order: %d" % (par_name, fft_order)
                )

                axs[par_i].scatter(
                    month_doys,
                    monthly_params[:, par_i],
                    marker="x",
                    facecolor="k",
                )

                self._set_monthly_ticks(axs[par_i])
        return fig, axs


class SlidingDistHourly(SlidingDist, seasonal.Torus):
    """Estimates parametric distributions for time series exhibiting
    seasonalities in daily cycles."""

    def __init__(
        self,
        distribution,
        data,
        dtimes,
        doy_width=5,
        hour_neighbors=4,
        fft_order=5,
        *args,
        **kwds,
    ):
        super(SlidingDistHourly, self).__init__(
            distribution,
            data,
            dtimes,
            doy_width=doy_width,
            fft_order=fft_order,
            kill_leap=True,
            *args,
            **kwds,
        )
        seasonal.Torus.__init__(self, hour_neighbors)
        # for property caching
        self._doy_hour_weights = None

    def plot_fourier_fit(self):
        fig, axes = plt.subplots(self.sliding_pars.shape[0], sharex=True)
        for par_i, ax in enumerate(axes):
            ax.plot(self.doys_unique, self.sliding_pars[par_i])
            ax.set_title(self.dist.parameter_names[par_i])
        try:
            plt.suptitle(self.dist.name)
        except AttributeError:
            pass
        return fig, axes

    @property
    def sliding_pars(self):
        if self._sliding_pars is None:
            n_pars = len(self.dist.parameter_names) - len(
                list(self.dist._clean_kwds(self.fixed_pars).keys())
            )
            self._sliding_pars = np.ones((self.n_doys, n_pars))
            for doy_ii, doy in tqdm(
                enumerate(self.doys_unique), disable=(not self.verbose)
            ):
                data = self.torus[self._torus_slice(doy)].ravel()
                fixed = {
                    par_name: func(self.doys_unique[doy_ii])
                    for par_name, func in list(self.fixed_pars.items())
                }
                if self.dist.isscipy or isinstance(
                    self.dist, distributions.Rain
                ):
                    self._sliding_pars[doy_ii] = self.dist.fit(data, **fixed)
                else:

                    def fit_full(x0):
                        return self.dist.fit_ml(
                            data,
                            weights=self.doy_hour_weights,
                            x0=x0,
                            method="Powell",
                            **fixed,
                        )

                    if doy_ii > 24 + self.hour_neighbors:
                        x0 = np.mean(
                            self._sliding_pars[
                                doy_ii
                                - 24
                                - self.hour_neighbors : doy_ii
                                - 24
                                + self.hour_neighbors
                            ],
                            axis=0,
                        )
                    else:
                        x0 = self.dist.fit(data, **fixed)
                    if np.nan in x0:
                        x0 = self.dist.fit(data, **fixed)

                    result = fit_full(x0)
                    if not result.success:
                        x0 = self.dist.fit(data, **fixed)
                        result = fit_full(x0)

                    # if not result.success:
                    #     dist = self.dist(*(list(x0) +
                    #                        [fixed[par_name]
                    #                         for par_name
                    #                         in self.dist.parameter_names
                    #                         if par_name in fixed]))
                    #     dist.plot_fit(data)
                    #     plt.show()

                    self._sliding_pars[doy_ii] = (
                        result.x if result.success else [np.nan] * len(x0)
                    )

            if self.verbose:
                print()

        # # try to interpolate over bad fittings
        # pars = self._sliding_pars
        # for par_i, par in enumerate(pars.T):
        #     if np.any(np.isnan(par)):
        #         half = len(par) / 2
        #         par_pad = np.concatenate((par[-half:], par, par[:half]))
        #         interp = my.interp_nan(par_pad)[half:-half]
        #         self._sliding_pars[:, par_i] = interp

        return self._sliding_pars.T

    @property
    def solution(self):
        if self._solution is None:
            self._solution = self.sliding_pars
        # if self._solution is None:
        #     trans = []
        #     for sl_par in self.sliding_pars:
        #         hours = np.array(sl_par.size / 24 * range(24))
        #         doys = self.doys_unique
        #         years = np.zeros_like(hours)
        #         par_torus = self.torus_fft(sl_par, hours, doys, years,
        #                                    fft_order=None)
        #         trans += [par_torus]
        #     self._solution = np.array(trans)
        return self._solution

    @solution.setter
    def solution(self, trans):
        self._solution = trans

    def trig2pars(self, parameters, _T=None, fft_order=None):
        fft_order = self.fft_order if fft_order is None else fft_order
        if _T is None:
            try:
                _T = self._T
            except AttributeError:
                _T = self._T = (2 * np.pi / 365 * self.doys)[np.newaxis, :]

        doys = np.atleast_1d(365 * np.squeeze(_T) / (2 * np.pi))
        # # doys_ii = np.where(my.isclose(self.doys_unique, doys[:, None]))[1]
        # # use the parameters of 28. feb for 29.feb
        # year_end_ii = np.where(np.diff(doys) < 0)[0] + 1
        # year_end_ii = np.concatenate(([0], year_end_ii, [len(doys)]))
        # for start_i, end_i in zip(year_end_ii[:-1], year_end_ii[1:]):
        #     year_slice = slice(start_i, end_i)
        #     year_doys = doys[year_slice]
        #     if np.max(year_doys) > 366:
        #         year_doys[year_doys > 31 + 29] -= 1
        #         doys[year_slice] = year_doys

        # doys_ii = ((doys - 1) / self.dt).astype(int)
        # if len(doys_ii) < len(doys):
        #     doys_ii = [my.val2ind(self.doys_unique, doy) for doy in doys]

        doys_ii = self.doys2doys_ii(doys)
        return parameters[:, doys_ii]
        # hour_dim_size = 24 + 2 * self.hour_neighbors
        # doy_dim_size = len(self.doys_unique)
        # padded_shape = hour_dim_size, doy_dim_size
        # pars = []
        # for omega in omegas:
        #     nth_largest = np.sort(np.ravel(np.abs(omega)))[-fft_order]
        #     omega[np.abs(omega) < nth_largest] = 0
        #     data_2d = np.fft.irfft2(omega, s=padded_shape)
        #     pars += [data_2d.T.ravel()[:_T.size]]
        # return pars

    def fourier_approx(self, fft_order=None):
        _T = (2 * np.pi / 365 * self.doys_unique)[None, :]
        return self.trig2pars(self.solution, _T=_T, fft_order=fft_order)


if __name__ == "__main__":
    import os
    import vg

    met_vg = vg.VG(
        ("R", "theta", "ILWR"),
        # rain_method="regression",
        rain_method="distance",
        # refit="R",
        # refit=True,
        verbose=True,
        dump_data=False,
    )
    met_vg.fit(p=3, seasonal=True)
    theta_incr = 1
    simt, sim = met_vg.simulate(
        theta_incr=theta_incr,
        primary_var="R",
        phase_randomize=True,
        phase_randomize_vary_mean=False,
    )
    prim_i = met_vg.primary_var_ii[0]
    data_mean = np.mean(met_vg.data_raw[prim_i]) / 24

    #     # conf = vg.config_konstanz_disag
#     import config_konstanz_disag as conf
#     from vg.core import vg_plotting

#     vg.set_conf(conf)
#     # from scipy.stats import distributions as sp_dists
#     dt_hourly, met = vg.read_met(os.path.join(vg.conf.data_dir, vg.conf.met_file))
#     rh = met["rh"]
#     finite_mask = np.isfinite(rh)
#     rh = rh[finite_mask]
#     dt_hourly = dt_hourly[finite_mask]
#     dist = vg.distributions.Censored(vg.distributions.norm)
#     rh_dist = SlidingDistHourly(
#         dist, rh, dt_hourly, fixed_pars=vg.conf.par_known_hourly["rh"], verbose=True
#     )
#     solution = rh_dist.fit()
#     # rh_dist.plot_monthly_fit(solution)
#     # rh_dist.plot_monthly_params()
#     rh_dist.plot_fourier_fit()
#     rh_dist.scatter_pdf()
#     #    for order in range(5):
#     #        theta_dist.plot_fourier_fit(order)
#     #    print theta_dist.chi2_test()
#     plt.show()
# #    rh, dtimes = vg.my.sumup(met["Qsw"], 24, dtimes_)
# #    rh_dist = SlidingDist(vg.distributions.beta, rh, dtimes, 15, 10,
# #                          fixed_pars=vg.conf.par_known["Qsw"])
# #    rh_dist.fit()
# # rh_dist.plot_fourier_fit(order=10)
# # rh_dist.scatter_pdf()


# #    vg.plt.plot(qsw_dist.sliding_pars)
# #    qsw_dist.plot_seasonality_fit()
# #    vg.plt.show()
