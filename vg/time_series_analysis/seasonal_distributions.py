"""Tools to fit seasonal distributions by describing the yearly cycle of
distribution parameters with triangular functions."""
from __future__ import division, print_function

import collections
from builtins import range, zip

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from past.utils import old_div
from scipy import stats

# import vg.helpers as my
from .. import helpers as my
from vg import smoothing, times
from vg.time_series_analysis import (distributions, optimize,
                                     seasonal)
from tqdm import tqdm


class SeasonalDist(seasonal.Seasonal):
    def __init__(self, distribution, x, datetimes, fixed_pars=None,
                 par_ntrig=None, time_form="%m", verbose=False,
                 kill_leap=False, **kwds):
        """distribution should be an object that implements pdf, cdf and so on.
        fixed_pars is expected to be a dictionary mapping parameter names to
        functions that take days of the year as input to calculate distribution
        parameters.
        time_form (e.g. "%m") determines what is used to generate a starting
        solution of trigonometric parameters.
        var_ntrig should be a sequence mapping distribution parameters to
        number of trigonometric parameters. If that is None, 3 trig  parameters
        per dist parameter are assumed
        """
        # super(SeasonalDist, self).__init__(x, datetimes, kill_leap=kill_leap)
        seasonal.Seasonal.__init__(self, x, datetimes, kill_leap=kill_leap)
        self.verbose = verbose
        # check if this is a generalized truncated distribution
        try:
            self.dist = distribution[0](distribution[1], **kwds)
        except TypeError:
            self.dist = distribution
        self.fixed_pars = {} if fixed_pars is None else fixed_pars
        if not hasattr(self.dist, "parameter_names"):
            # we assume that we have come across a scipy.stats.distribution
            self.dist.parameter_names = ["loc", "scale"]
            if self.dist.shapes is not None:
                self.dist.parameter_names += self.dist.shapes.split(",")
            self.dist.__bases__ = distributions.MyDist
            self.dist.scipy_ = True
        if par_ntrig is None:
            self.par_ntrig = [3] * len(self.dist.parameter_names)
        else:
            self.par_ntrig = par_ntrig
        self.time_form = time_form

        self.non_fixed_names = [par_name for par_name
                                in self.dist.parameter_names
                                if par_name
                                not in self.fixed_pars.keys()]
        if self.dist.supplements_names is not None:
            for suppl in self.dist.supplements_names:
                self.non_fixed_names.remove(suppl)

        self.n_trig_pars = 3
        self.pre_start = []
        self._supplements = None

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
        a, b_0, phi = \
            np.array(trig_pars).reshape(-1, 3).T[:, :, np.newaxis]
        # i lied, this took me some time. but now, the parameters are vertical
        # to T_ and we are broadcasting like some crazy shit pirate radio
        # station
        return a + b_0 * np.sin(_T + phi)

    def fixed_values_dict(self, doys=None):
        """This holds a cache for the self.doys values. If doys is given, those
        are used and the cache is left untouched."""
        def build_dict(doys):
            return dict((fixed_par_name, func(doys))
                        for fixed_par_name, func in self.fixed_pars.items())
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
        params = {key: val for key, val in
                  list(self.fixed_values_dict(doys).items())}
        if doys is not None:
            doys = np.copy(doys) * 2 * np.pi / 365
        params.update(
            {param_name: values for param_name, values
             in zip(self.non_fixed_names,
                    self.trig2pars(trig_pars, _T=doys))})
        if self.supplements is not None:
            params.update({param_name: values for param_name, values
                           in self.all_supplements_dict(doys).items()})
        return params

    def all_supplements_dict(self, doys=None):
        """Dictionary of all supplements assembled by doy distance."""
        if doys is not None:
            doys = np.copy(doys) * 2 * np.pi / 365
        else:
            doys = self.doys
        doys_ii = self.doys2doys_ii(doys)
        supplement_names = self.dist.supplements_names
        return {name:
                np.array([self.supplements[doy_ii][name]
                          for doy_ii in doys_ii])
                for name in supplement_names}

    @property
    def monthly_grouped(self):
        try:
            return self._monthly_grouped
        except AttributeError:
            self._monthly_grouped = times.time_part_sort(self.datetimes,
                                                         self.data,
                                                         self.time_form)
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
            self._daily_grouped_x = \
                times.time_part_sort(self.datetimes, self.data, "%j")[1]
        return self.daily_grouped_x

    @property
    def monthly_params(self):
        """These are the distribution parameters per month. Returns a
        12xn_distribution_parameters array."""
        return np.array([self.dist.fit(
            sub_x,
            **dict((par_name,
                    np.average(func(self.doys[times.time_part(self.datetimes,
                                                              self.time_form)
                                              == month])))
                   for par_name, func in list(self.fixed_pars.items())))
            for month, sub_x in zip(*self.monthly_grouped)])

    @property
    def monthly_supplements(self):
        pass

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
            first_doms = times.str2datetime(["01.%02d.2000 00:00:00" % month
                                             for month in range(1, 13)])
            _T = np.array(times.time_part(first_doms, "%j"), dtype=float)
        _T *= 2 * np.pi / 365

        def error(trig_pars, emp_pars, times_=None):
            times_ = _T if times_ is None else times_
            return np.sum((self.trig2pars(trig_pars, times_) - emp_pars) ** 2)

        start_pars = []
        for par_name, par_timeseries in zip(self.dist.parameter_names,
                                            self.monthly_params.T):
            # if the distribution comes with lower and upper bounds, find a
            # solution for those that contains all the measurement points.
            if par_name == "l" and "l" not in self.fixed_pars:
                minima = np.array([var.min()
                                   for var in self.daily_grouped_x])
                smoothed_mins = smoothing.smooth(minima, 365, periodic=True)
                # place the phase shift so that the minimum of the sine is at
                # the the minimum of the dataset
                mint = self.doys[np.argmin(smoothed_mins)]
                phi0 = 1.5 * np.pi - mint * 2 * np.pi / 365
                b_00 = .5 * (smoothed_mins.max() - smoothed_mins.min())
                a0 = smoothed_mins.mean()
                start = [a0, b_00, phi0] + self.pre_start
                lowest_diff = np.min(smoothed_mins - minima)
                k = 1.5
                while np.any(smoothed_mins < k * lowest_diff):
                    k += .5
                fit_me = smoothed_mins - k * lowest_diff
                times_ = np.arange(366.) * 2 * np.pi / 365
                start_pars += list(opt_func(error, start,
                                            args=(fit_me, times_), disp=False,
                                            **kwds))
            elif par_name == "u" and "u" not in self.fixed_pars:
                maxima = np.array([var.max()
                                   for var in self.daily_grouped_x])
                smoothed_maxs = smoothing.smooth(maxima, 365, periodic=True)
                maxt = self.doys[np.argmax(smoothed_maxs)]
                phi0 = .5 * np.pi - maxt * 2 * np.pi / 365
                b_00 = .5 * (smoothed_maxs.max() - smoothed_maxs.min())
                a0 = smoothed_maxs.mean()
                start = [a0, b_00, phi0] + self.pre_start
                highest_diff = np.max(maxima - smoothed_maxs)
                fit_me = smoothed_maxs + 1.2 * highest_diff
                times_ = np.arange(366.) * 2 * np.pi / 365
                start_pars += list(opt_func(error, start, disp=False,
                                            args=(fit_me, times_), **kwds))
            elif par_name not in self.fixed_pars:
                start_pars += list(opt_func(error, x0, args=(par_timeseries,),
                                            disp=False, **kwds))
        self._start_pars = start_pars
        return start_pars

    def _complete_dist_call(self, func, trig_pars, x=None, doys=None,
                            broadcast=False, **kwds):
        """Abstracts the common ground for self.{pdf,cdf,ppf} which call the
        according functions of the underlying distribution."""
        params = self.all_parameters_dict(trig_pars, doys)
        params.update(kwds)
        x = self.data if x is None else x
        if broadcast:
            xx = x[np.newaxis, :]
            params = dict((par_name, val[:, np.newaxis])
                          for par_name, val in list(params.items()))
            if self.dist.scipy_:
                args = [params[par_name] for par_name
                        in self.dist.parameter_names]
                return func(xx, *args)
            return func(xx, **params)
        else:
            if self.dist.scipy_:
                args = [params[par_name] for par_name
                        in self.dist.parameter_names]
                return np.vectorize(func)(x, *args)
            return func(x, **params)

    def pdf(self, trig_pars, x=None, doys=None, **kwds):
        return self._complete_dist_call(self.dist.pdf, trig_pars, x, doys,
                                        **kwds)

    def cdf(self, trig_pars, x=None, doys=None, **kwds):
        return self._complete_dist_call(self.dist.cdf, trig_pars, x, doys,
                                        **kwds)

    def ppf(self, trig_pars, quantiles=None, doys=None, **kwds):
        return self._complete_dist_call(self.dist.ppf, trig_pars, quantiles,
                                        doys, **kwds)

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

    def fit(self, x=None, opt_func=optimize.simulated_annealing, x0=None,
            **kwds):
        if x is not None:
            self.data = x
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
            quantiles = self.cdf(trig_pars)
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
        result = opt_func(unlikelihood, x0, constraints=(constraints,),
                          callback=lambda x: setattr(self, "_solution", x),
                          **kwds)
#        try:
#            result = opt_func(unlikelihood, x0, constraints=(constraints,),
#                              callback=lambda x: setattr(self, "_solution",
#                                                         x),
#                              **kwds)
#        except ValueError:
#            #HACK!
#            result = x0
        if opt_func is sp.optimize.anneal:
            print("retval was %d" % result[-1])
            result = result[0]
        self._solution = result
        return result

    def chi2_test(self, k=None):
        """Chi-square goodness-of-fit test.
        H0: The given data **x** follows **distribution** with parameters
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
        quantiles = self.cdf(self.solution)
        n = len(quantiles)
        n_parameters = len(self.dist.parameter_names)
        if k is None:
            # k = int(n ** .5)
            k = n_parameters + 2
        observed = np.histogram(quantiles[np.isfinite(quantiles)], k)[0]
        expected = old_div(float(n), k)
        chi_test = np.sum(old_div((observed - expected) ** 2, expected))
        # degrees of freedom:
        dof = k - n_parameters - 1
        return stats.chisqprob(chi_test, dof)

    def scatter_cdf(self, trig_pars=None, figsize=None, title=None, *args,
                    **kwds):
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
        plt.scatter(self.doys, self.data, marker="o",
                    facecolors=(0, 0, 0, 0), edgecolors=(0, 0, 0, .5))
        # plot lower and upper bounds
        for fixed_values in list(self.fixed_values_dict(doys).values()):
            plt.plot(doys, fixed_values, "r")
        try:
            for par in (trig_pars[2 * self.n_trig_pars:
                                  3 * self.n_trig_pars],
                        trig_pars[3 * self.n_trig_pars:
                                  4 * self.n_trig_pars]):
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

    def scatter_pdf(self, trig_pars=None, figsize=None, title=None,
                    opacity=.25, s_kwds=None, *args, **kwds):
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
        # dens[dens < 1e-12] = 0
        fig = plt.figure(figsize=figsize)
        ax1 = fig.gca()
        ax1.contourf(self.doys_unique, xx, dens.T, 15)
#        plt.colorbar(co)
        plt.scatter(self.doys, self.data, facecolors=(0, 0, 0, 0),
                    edgecolors=(0, 0, 0, opacity), **s_kwds)
#        if self.dist in (distributions.beta, distributions.kumaraswamy,
#                         distributions.Kumaraswamy,
#                         distributions.weibull, distributions.logitnormal,
#                         distributions.lognormallu, distributions.rayleighlu):
        # plot lower and upper bounds
        for fixed_values in self.fixed_values_dict(self.doys_unique).values():
            plt.plot(self.doys_unique, fixed_values, "r")
        try:
            for par in (trig_pars[2 * self.n_trig_pars:
                                  3 * self.n_trig_pars],
                        trig_pars[3 * self.n_trig_pars:
                                  4 * self.n_trig_pars]):
                plt.plot(self.doys_unique,
                         np.squeeze(self.trig2pars(par, _T)), "r")
        except (IndexError, ValueError):
            pass

#        if self.dist is distributions.norm:
#            # do not plot the mean (boring!)
#            ax = plt.gca().twinx()
#            par = self.trig2pars(trig_pars[self.n_trig_pars:
#                                           2 * self.n_trig_pars], _T)
#            par = par.reshape(doys.shape)
#            plt.plot(doys, par, "r", label=self.dist.parameter_names[1])
#        else:
        # plot the first two distribution parameters on a different yscale

#        ax = plt.gca().twinx()
#        for par_i, par in enumerate((trig_pars[:self.n_trig_pars],
#                                     trig_pars[self.n_trig_pars:
#                                               2 * self.n_trig_pars])):
#            ax.plot(doys,
#                    self.trig2pars(par, _T).reshape(doys.shape) / n_sumup,
#                    label=self.dist.parameter_names[par_i])
        plt.xlim(0, 366)
        plt.ylim(xx.min(), xx.max())
        plt.xticks((1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335),
                   ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                    'Sep', 'Oct', 'Nov', 'Dec'), rotation=45)
#        plt.xlabel("Day of Year")
        plt.grid()
#        plt.legend()
        if title is not None:
            plt.title(title)
        # plt.tight_layout()
        return fig, ax1

    def plot_seasonality_fit(self, solution=None):
        if self.time_form == "%m":
            doys = np.array(
                times.time_part(
                    times.str2datetime(["2011 %d" % month
                                        for month in range(1, 13)],
                                       "%Y %m"),
                    "%j"), dtype=float)
        elif self.time_form == "%W":
            doys = \
                times.datetime2doy(
                    times.str2datetime(["2011 %d" % week
                                        for week in range(0, 54)],
                                       "%Y %W"))

        _T = 2 * np.pi * doys / 365
        if solution is None:
            solution = self.start_pars
        dist_params = self.trig2pars(solution, _T)
        for par_name, emp_pars, fit_pars in zip(self.dist.parameter_names,
                                                self.monthly_params.T,
                                                dist_params):
            plt.figure()
            plt.plot(emp_pars, label="empirical")
            plt.plot(fit_pars, label="fitted")
            plt.title(par_name)
            plt.legend()
        plt.show()

    def plot_monthly_fit(self, solution=None, n_classes=30, dists_alt=None):
        if solution is None:
            monthly_params = self.monthly_params
        else:
            # get the doys of the middle of the months very roughly
            month_doys = np.linspace(15, 367, 12,
                                     endpoint=False).astype(np.int)
            # get a parameter set per month
            monthly_params_dict = \
                self.all_parameters_dict(solution, month_doys)
            monthly_params = my.list_transpose([monthly_params_dict[par_name]
                                                for par_name in
                                                self.dist.parameter_names])
            
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(23, 12))
        ax = axes.ravel()
        plt.suptitle(self.dist.name)
        fig.canvas.set_window_title(self.dist.name)
        for ii, values in enumerate(self.monthly_grouped_x):
            ax1 = ax[ii]

            # the histogram of the data
            bins = ax1.hist(values, n_classes, density=True, facecolor='grey',
                            alpha=0.75)[1]

            class_middles = 0.5 * (bins[1:] + bins[:-1])
            if self.dist.scipy_:
                density = self.dist.pdf(class_middles, *monthly_params[ii])
            else:
                monthly_dict = dict((par_name, pars) for par_name, pars in
                                    zip(self.dist.parameter_names,
                                        monthly_params[ii]))
                density = self.dist.pdf(class_middles, **monthly_dict)
            ax1.plot(class_middles, density, 'r--')

            # the quantile part
            ax2 = ax1.twinx()
            # empirical cdf
            values_sort = np.sort(values)
            ranks_emp = (.5 + np.arange(len(values))) / len(values)
            ax2.plot(values_sort, ranks_emp)
            # theoretical cdf
            xx = np.linspace(values.min(), values.max(), 5e2)
            if self.dist.scipy_:
                ranks_theory = self.dist.cdf(xx, *monthly_params[ii])
            else:
                ranks_theory = self.dist.cdf(xx, **monthly_dict)
            ax2.plot(xx, ranks_theory, 'r--')
            if hasattr(self.dist, "f_thresh"):
                ax2.axvline(self.dist.f_thresh, linestyle="--",
                            linewidth=1, color="gray")

            if dists_alt:
                if not isinstance(dists_alt, collections.Iterable):
                    dists_alt = dists_alt,

                for dist_alt in dists_alt:
                    dist = dist_alt(*dist_alt.fit(values))
                    ax1.plot(class_middles, dist.pdf(class_middles), "--")
                    ax2.plot(xx, dist.cdf(xx), "--")

            params_str = (", "
                          .join(" %s: %.3f" % (par_name, par)
                                for par_name, par in
                                zip(self.dist.parameter_names,
                                    monthly_params[ii])
                                if (self.dist.supplements_names is not None
                                    and par_name
                                    not in self.dist.supplements_names)))
            plt.title("month:%d %s" % (ii + 1, params_str),
                      fontsize=11)
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
        plt.tight_layout()
        return fig

    def plot_monthly_params(self):
        for par_name, values in zip(self.dist.parameter_names,
                                    self.monthly_params.T):
            plt.figure()
            plt.plot(values)
            plt.title(par_name)
        plt.show()


class SlidingDist(SeasonalDist):
    """Can we get a better picture at the seasonalities of the distribution
    parameters if we estimate them over a sliding window over the doys?"""
    def __init__(self, distribution, x, dtimes, doy_width=15, fft_order=4,
                 solution=None, supplements=None, *args, **kwds):
        super(SlidingDist, self).__init__(distribution, x, dtimes, *args,
                                          **kwds)
        self.doy_width, self.fft_order = doy_width, fft_order
        # usefull to assess goodness-of-fit/overfitting
        self.n_trig_pars = 2 * fft_order

        self.solution = solution
        self.supplements = supplements
        self._doy_mask = self._sliding_pars = None

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

    def doys2doys_ii(self, doys):
        """Doys indices from doy values."""
        # doys_ii = np.where(my.isclose(self.doys_unique, doys[:, None]))[1]
        # use the parameters of 28. feb for 29.feb
        year_end_ii = np.where(np.diff(doys) < 0)[0] + 1
        year_end_ii = np.concatenate(([0], year_end_ii, [len(doys)]))
        for start_i, end_i in zip(year_end_ii[:-1], year_end_ii[1:]):
            year_slice = slice(start_i, end_i)
            year_doys = doys[year_slice]
            if np.max(year_doys) > 366:
                year_doys[year_doys > 31 + 29] -= 1
                doys[year_slice] = year_doys

        doys_ii = ((doys - 1) / self.dt).astype(int)
        if len(doys_ii) < len(doys):
            doys_ii = [my.val2ind(self.doys_unique, doy) for doy in doys]
        return doys_ii

    @property
    def sliding_pars(self):
        if self._sliding_pars is None:
            n_pars = (self.dist.n_pars -
                      len(list(self.dist._clean_kwds(self.fixed_pars).keys())))
            self._sliding_pars = np.ones((self.n_doys, n_pars))
            if self.dist.supplements_names:
                # we need to save supplements also (needed for method
                # calling, but not fitted)
                self._supplements = []
            for doy_ii in tqdm(range(self.n_doys), disable=(not self.verbose)):
                data = self.data[self.doy_mask[doy_ii]]
                if not self.dist.scipy_:
                    # weight the data by doy-distance
                    doys = self.doys[self.doy_mask[doy_ii]]
                    doy_dist = times.doy_distance(doy_ii + 1, doys)
                    weights = (1 - doy_dist /
                               (self.doy_width + 2)) ** 2
                fixed = {par_name: func(self.doys_unique[doy_ii])
                         for par_name, func in list(self.fixed_pars.items())}
                if self.dist.scipy_ or \
                   isinstance(self.dist, distributions.Rain):
                    self._sliding_pars[doy_ii] = self.dist.fit(data, **fixed)
                else:
                    x0 = (self._sliding_pars[doy_ii - 1]
                          if doy_ii > 0
                          else self.dist.fit(data, **fixed))
                    # result = self.dist.fit(data, weights=weights, x0=x0,
                    #                        method='Powell', **fixed)
                    result = self.dist.fit_ml(data, weights=weights, x0=x0,
                                              method='Powell', **fixed)
                    self._sliding_pars[doy_ii] = \
                        result.x if result.success else [np.nan] * len(x0)
                    if result.supplements:
                        self._supplements += [result.supplements]
            if self.verbose:
                print()

        # try to interpolate over bad fittings
        pars = self._sliding_pars
        for par_i, par in enumerate(pars.T):
            if np.any(np.isnan(par)):
                half = len(par) // 2
                par_pad = np.concatenate((par[-half:], par, par[:half]))
                interp = my.interp_nan(par_pad)[half:-half]
                self._sliding_pars[:, par_i] = interp

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

    def fourier_approx_new(self, fft_order=4, trig_pars=None):
        if trig_pars is None:
            trig_pars = self.solution

        _fourier_approx = \
            np.empty((len(self.dist.parameter_names), self.n_doys))
        for ii, trans_par in enumerate(trig_pars):
            # find the fft_order biggest amplitudes
            ii_below = \
                np.argsort(np.abs(trans_par))[:len(trans_par) - fft_order - 1]
            pars = np.copy(trans_par)
            pars[ii_below] = 0
            _fourier_approx[ii] = np.fft.irfft(pars, self.n_doys)

        return _fourier_approx

    def fourier_approx(self, fft_order=4, trig_pars=None):
        if trig_pars is None:
            trig_pars = self.solution

        _fourier_approx = \
            np.empty((len(self.dist.parameter_names), self.n_doys))
        for ii, trans_par in enumerate(trig_pars):
            _fourier_approx[ii] = \
                np.fft.irfft(trans_par[:fft_order + 1], self.n_doys)
        return _fourier_approx

    def fit(self, x=None, **kwds):
        if x is not None:
            self.data = x
        return self.solution

    def trig2pars(self, trig_pars, _T=None, fft_order=None):
        fft_order = self.fft_order if fft_order is None else fft_order
        if _T is None:
            try:
                _T = self._T
            except AttributeError:
                _T = self._T = (2 * np.pi / 365 * self.doys)[np.newaxis, :]
        doys = np.atleast_1d(365 * np.squeeze(_T) / (2 * np.pi))
        doys_ii = np.where(np.isclose(self.doys_unique, doys[:, None]))[1]
        if len(doys_ii) < len(doys):
            doys_ii = [my.val2ind(self.doys_unique, doy) for doy in doys]
        fourier_pars = self.fourier_approx(fft_order, trig_pars)
        return np.array([fourier_pars[:, doy_i] for doy_i in doys_ii]).T

    def plot_fourier_fit(self, fft_order=None):
        """Plots the Fourier approximation of all parameters."""
        fft_order = self.fft_order if fft_order is None else fft_order
        fig, axes = plt.subplots(len(self.non_fixed_names), sharex=True,
                                 squeeze=True)
        pars = self.fourier_approx_new(fft_order)
        if pars.shape[1] > 1:
            for par_i, par_name in enumerate(self.non_fixed_names):
                axes[par_i].plot(self.doys_unique, self.sliding_pars[par_i])
                axes[par_i].plot(self.doys_unique,
                                 self.fourier_approx_new(fft_order)[par_i],
                                 label="new")
                axes[par_i].plot(self.doys_unique,
                                 self.fourier_approx(fft_order)[par_i],
                                 label="old")
                axes[par_i].grid(True)
                axes[par_i].set_title("%s Fourier fft_order: %d" %
                                      (par_name, fft_order))
        plt.legend(loc="best")
        return fig, axes


class SlidingDistHourly(SlidingDist, seasonal.Torus):

    """Estimates parametric distributions for time series exhibiting
    seasonalities in daily cycles."""

    def __init__(self, distribution, data, dtimes, doy_width=5,
                 hour_neighbors=4, fft_order=5, *args, **kwds):
        super(SlidingDistHourly, self).__init__(distribution, data,
                                                dtimes,
                                                doy_width=doy_width,
                                                fft_order=fft_order,
                                                kill_leap=True, *args,
                                                **kwds)
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
            n_pars = (len(self.dist.parameter_names) -
                      len(list(self.dist._clean_kwds(self.fixed_pars).keys())))
            self._sliding_pars = np.ones((self.n_doys, n_pars))
            for doy_ii, doy in tqdm(enumerate(self.doys_unique),
                                    disable=(not self.verbose)):
                data = self.torus[self._torus_slice(doy)].ravel()
                fixed = {par_name: func(self.doys_unique[doy_ii])
                         for par_name, func in list(self.fixed_pars.items())}
                if self.dist.scipy_ or \
                   isinstance(self.dist, distributions.Rain):
                    self._sliding_pars[doy_ii] = self.dist.fit(data, **fixed)
                else:
                    def fit_full(x0):
                        return self.dist.fit_ml(data,
                                                weights=self.doy_hour_weights,
                                                x0=x0, method='Powell',
                                                **fixed)
                    if doy_ii > 24 + self.hour_neighbors:
                        x0 = np.mean(self._sliding_pars[doy_ii - 24 -
                                                        self.hour_neighbors:
                                                        doy_ii - 24 +
                                                        self.hour_neighbors],
                                     axis=0)
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

                    self._sliding_pars[doy_ii] = \
                        result.x if result.success else [np.nan] * len(x0)

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

    def fourier_approx_new(self, fft_order=None):
        return self.fourier_approx(fft_order=fft_order)


if __name__ == "__main__":
    import os
    import vg
    conf = vg.config_konstanz_disag
    from vg.core import vg_plotting
    vg.conf = vg.vg_base.conf = vg_plotting.conf = conf
    # from scipy.stats import distributions as sp_dists
    dt_hourly, met = vg.read_met(os.path.join(vg.conf.data_dir,
                                              vg.conf.met_file))
    rh = met["rh"]
    finite_mask = np.isfinite(rh)
    rh = rh[finite_mask]
    dt_hourly = dt_hourly[finite_mask]
    dist = vg.distributions.Censored(vg.distributions.norm)
    rh_dist = SlidingDistHourly(dist,
                                rh,
                                dt_hourly,
                                fixed_pars=vg.conf.par_known_hourly["rh"],
                                verbose=True)
    solution = rh_dist.fit()
    # rh_dist.plot_monthly_fit(solution)
    # rh_dist.plot_monthly_params()
    rh_dist.plot_fourier_fit()
    rh_dist.scatter_pdf()
#    for order in range(5):
#        theta_dist.plot_fourier_fit(order)
#    print theta_dist.chi2_test()
    plt.show()
#    rh, dtimes = vg.my.sumup(met["Qsw"], 24, dtimes_)
#    rh_dist = SlidingDist(vg.distributions.beta, rh, dtimes, 15, 10,
#                          fixed_pars=vg.conf.par_known["Qsw"])
#    rh_dist.fit()
    # rh_dist.plot_fourier_fit(order=10)
    # rh_dist.scatter_pdf()


#    vg.plt.plot(qsw_dist.sliding_pars)
#    qsw_dist.plot_seasonality_fit()
#    vg.plt.show()
