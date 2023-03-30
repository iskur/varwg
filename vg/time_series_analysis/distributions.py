import os
import sys
import subprocess
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sp_optimize
from future import standard_library
from future.backports import urllib
from future.utils import with_metaclass
from past.utils import old_div
from scipy import interpolate, linalg, special, stats
from scipy.integrate import cumtrapz


try:
    import numexpr as ne
    NE = True
except ImportError:
    NE = False

from vg import helpers as my
from vg.time_series_analysis import _kde as kde

standard_library.install_aliases()

try:
    from vg.time_series_analysis import owens
except ImportError:
    if sys.platform != "win32":
        import urllib.request, urllib.error, urllib.parse
        import socket
        socket.setdefaulttimeout(10)
        warn = False
        url = "http://people.sc.fsu.edu/~jburkardt/f_src/owens/owens.f90"
        try:
            src_dir = os.path.dirname(__file__)
        except NameError:
            src_dir = os.path.abspath(".")
        # if os.path.exists(os.path.join(src_dir, "owens.f90")):
        #     warn = True
        if not warn:
            with my.chdir(src_dir):
                with open("owens.f90", "w") as owens_file:
                    try:
                        content_full = (urllib.request
                                        .urlopen(url)
                                        .read()
                                        .decode("utf-8"))
                        # under python 3: "character ( len = 9 )" causes
                        # f2py to fail, so we cut out the subroutine
                        # timestep, which is not needed for owens-t
                        content = []
                        keep_line = True
                        for line in content_full.split("\n"):
                            if line.startswith("subroutine timestamp"):
                                keep_line = False
                            if keep_line:
                                content += [line]
                            if not keep_line and line == "end":
                                keep_line = True
                        content = os.linesep.join(content)
                        owens_file.write(content + os.linesep)
                    except urllib.error.URLError:
                        warn = True
                try:
                    subprocess.call("f2py -L$PREFIX$/lib -c -m owens owens.f90",
                                    shell=True)
                    from vg.time_series_analysis import owens
                    warn = False
                except:
                    warn = True
        if warn:
            warnings.warn("""Could not import owens t function.
                Try (if on linux):
                wget {url}
                f2py -c -m owens owens.f90
                cp owens.so {src_dir}/
                """.format(url=url, src_dir=src_dir))
            owens = False
    else:
        owens = False
if owens:
    owens_t = np.vectorize(lambda h, a: owens.t(h, a))

# some special functions vectorized to be able to handle arrays as input
gamma_func = np.vectorize(lambda x: special.gamma(x))
gammaln = np.vectorize(lambda x: special.gammaln(x))
gammainc = np.vectorize(lambda a, x: special.gammainc(a, x))
gammaincinv = np.vectorize(lambda a, qq: special.gammaincinv(a, qq))
digamma = np.vectorize(lambda a: special.digamma(a))
hyp1f1 = np.vectorize(lambda a, b, x: special.hyp1f1(a, b, x))
nctdtr = np.vectorize(lambda x1, x2, x3: special.nctdtr(x1, x2, x3))
nctdtrit = np.vectorize(lambda x1, x2, x3: special.nctdtrit(x1, x2, x3))
stdtr = np.vectorize(lambda df, x: special.stdtr(df, x))
stdtrit = np.vectorize(lambda df, qq: special.stdtrit(df, qq))


def max_likelihood(density_func, x0, values=None, opt_func=None,
                 disp=False, weights=None, method="BFGS",
                 bounds=None, *args, **kwds):
    """Fits parameters of a density function to data, using the log-maximum-
    likelihood approach."""
    if opt_func is None:
        from scipy import optimize
        opt_func = optimize.minimize
    if weights is None and values is not None:
        weights = np.ones_like(values)
    # i hereby define the unlikelihood function as the negative
    # log-likelihood function, so that i can maximize the likelihood
    # by minimizing the unlikelihood
    if values is None:
        args = ()

        def unlikelihood(params):
            return -np.sum(np.log(weights * density_func(params) + 1e-12))
    else:
        args = values,

        def unlikelihood(params, values):
            # we have to express fixed values in terms of kwds,
            # because there could be collisions otherwise...

            # first come, first serve: interpret the first n elements
            # of params as the first n variables in
            # dist.parameter_names
            p_kwds = {
                par_name: value
                for par_name, value in zip(
                    density_func.__self__.parameter_names, params)
                if par_name not in kwds
            }
            p_kwds.update(kwds)
            dens = density_func(values, **p_kwds)
            try:
                inside = weights * dens
            except FloatingPointError:
                inside = dens
            inside[~np.isfinite(inside)] = 1e-9
            inside[inside <= 0] = 1e-9
            return -np.sum(np.log(inside))

    return opt_func(
        unlikelihood,
        x0,
        args=args,
        # bounds=density_func._bounds,
        bounds=bounds,
        # method=method,
        # method=("L-BFGS-B" if density_func._bounds else method),
        method=("L-BFGS-B" if bounds else method),
        options={"maxiter": 1e4 * len(x0),
                 "disp": disp})


def min_ks(cdf, x0, values, opt_func=None, disp=False, *args, **kwds):
    """Fits parameters of a cumulative distribution function to data,
    minimizing the Kolmogorov-Smirnof test statistic."""
    if opt_func is None:
        opt_func = sp_optimize.fmin
    n = len(values)
    values_sorted = np.sort(values)
    ranks_plus = old_div(np.arange(0., n), n)
    ranks_minus = old_div(np.arange(1., n + 1), n)

    def ks(params):
        # we have to express fixed values in terms of kwds, because there
        # could be collisions otherwise...

        # first come, first serve: interpret the first n elements of params
        # as the first n variables in dist.parameter_names
        p_kwds = {
            par_name: value
            for par_name, value in zip(cdf.__self__.parameter_names, params)
            if par_name not in kwds
        }
        p_kwds.update(kwds)

        cdf_values = cdf(values_sorted, **p_kwds)
        cdf_values[np.isnan(cdf_values)] = np.inf
        dmin_plus = np.abs(cdf_values - ranks_plus).max()
        dmin_minus = np.abs(cdf_values - ranks_minus).max()
        return max(dmin_plus, dmin_minus)

    return opt_func(ks, x0, args=args, maxiter=1e4 * len(x0),
                    maxfun=1e4 * len(x0), disp=disp)


def min_fsum(cdf, x0, values, opt_func=None, disp=False, *args, **kwds):
    if opt_func is None:
        opt_func = sp_optimize.fmin
    n = len(values)
    values_sorted = np.sort(values)
    ranks_plus = old_div(np.arange(0., n), n)
    ranks_minus = old_div(np.arange(1., n + 1), n)

    def ks(params):
        # we have to express fixed values in terms of kwds, because there
        # could be collisions otherwise...

        # first come, first serve: interpret the first n elements of params
        # as the first n variables in dist.parameter_names
        p_kwds = {
            par_name: value
            for par_name, value in zip(cdf.__self__.parameter_names, params)
            if par_name not in kwds
        }
        p_kwds.update(kwds)

        cdf_values = cdf(values_sorted, **p_kwds)
        cdf_values[np.isnan(cdf_values)] = np.inf
        dmin_plus = ((cdf_values - ranks_plus)**2).sum()
        dmin_minus = ((cdf_values - ranks_minus)**2).sum()
        return dmin_plus + dmin_minus

    return opt_func(ks, x0, args=args, maxiter=1e4 * len(x0),
                    maxfun=1e4 * len(x0), disp=disp)


class DistMeta(ABCMeta):
    """Assure some class attributes are set, so that life is easier later on.
    """
    # pylint does not let me call the first parameter meta :(
    def __new__(cls, name, bases, cls_dict):
        new_cls = super(DistMeta, cls).__new__(cls, name, bases, cls_dict)
        new_cls.name = name.lower()
        # store the distribution parameter names as a class attribute
        # take the pdf as an example
        # could do some signature checks on cdf and ppf with that...
        code = cls_dict["_pdf"].__code__
        # first varname is self, second is x
        varnames = list(code.co_varnames[2:code.co_argcount])
        # supplements are needed to call pdf, cdf and ppf, but are not
        # fitted (e.g. data for kernel density)
        # number of fittable parameters
        n_pars = len(varnames)
        if "supplements_names" in cls_dict:
            n_pars -= len(cls_dict["supplements_names"])
        else:
            new_cls.supplements_names = None
        if "_bounds" not in new_cls.__dict__:
            new_cls._bounds = None
        new_cls.parameter_names = tuple(varnames)
        new_cls.n_pars = n_pars
        return new_cls


class Dist(metaclass=DistMeta):
    """Mimics part of the interface of stats.rv_continuous. Comes with a few
    extra goodies."""
    # i would like to call the first item "solution", but in order to
    # have a nice compatibility to the optime.minimize result object,
    # it has to be "x"
    Result = namedtuple("Result", ("x", "supplements", "success"))
    isscipy = False

    @abstractmethod
    def _pdf(self):
        pass

    @abstractmethod
    def _cdf(self):
        pass

    @abstractmethod
    def _ppf(self):
        pass

    @abstractmethod
    def _fit(self):
        pass

    @property
    def scipy_(self):
        return False

    def _clean_kwds(self, kwds):
        """Return a copy with only the keywords that are also in
        self.parameter_names."""
        return {key: value
           for key, value in list(kwds.items())
           if key in self.parameter_names}

    def fit(self, *args, **kwds):
        return self._fit(*args, **self._clean_kwds(kwds))

    def sample(self, *args, **kwds):
        size = np.atleast_1d(args[0])
        qq = np.random.random(size)
        return self.ppf(qq, *args[1:], **self._clean_kwds(kwds))

    @my.asscalar
    def pdf(self, *args, **kwds):
        densities = np.atleast_1d(self._pdf(*args, **self._clean_kwds(kwds)))
        if "x" in kwds:
            x = kwds.pop("x")
            invalid_x = self._invalid_x(x, *args, **kwds)
        else:
            invalid_x = self._invalid_x(args[0], *args[1:], **kwds)
        return np.where(invalid_x, np.nan, densities)

    @my.asscalar
    def cdf(self, *args, **kwds):
        qq = np.atleast_1d(self._cdf(*args, **self._clean_kwds(kwds)))
        if "x" in kwds:
            x = kwds.pop("x")
            invalid_x = self._invalid_x(x, *args, **kwds)
        else:
            invalid_x = self._invalid_x(args[0], *args[1:], **kwds)
        return np.where(invalid_x, np.nan, qq)

    @my.asscalar
    def ppf(self, *args, **kwds):
        if "qq" in kwds:
            qq = kwds.pop("qq")
        else:
            qq = args[0]
        qq = np.atleast_1d(qq)
        finite_mask = np.isfinite(qq)
        quantiles_finite = np.where(finite_mask, qq, -1)
        x = np.atleast_1d(
            self._ppf(quantiles_finite, *args[1:],
                      **self._clean_kwds(kwds)))
        if quantiles_finite.size == 1:
            quantiles_finite = np.full_like(x, quantiles_finite)
        x[(quantiles_finite < 0) | (quantiles_finite > 1)] = np.nan
        return x

    def __call__(self, *params):
        return Frozen(self, *params)

    def fit_ml(self, values, x0=None, *args, **kwds):
        if x0 is None:
            x0 = (1, ) * len(self.parameter_names)
        result = max_likelihood(self.pdf, x0, values, bounds=self._bounds,
                                *args, **kwds)
        return self.Result(x=result.x,
                           supplements=None,
                           success=result.success)

    # def fit_ml(self, values, x0=None, *args, **kwds):
    #     if x0 is None:
    #         x0 = (1, ) * len(self.parameter_names)
    #     if self.supplements_names is not None:
    #         x0 = list(x0)
    #         for name in self.supplements_names:
    #             x0[self.parameter_names.index(name)] = None
    #             # x0.pop(self.parameter_names.index(name))
    #     params = max_likelihood(self.pdf, x0, values, *args, **kwds)
    #     if self.supplements_names is not None:
    #         supplements = {name: params[self.parameter_names.index(name)]
    #                        for name in self.supplements_names}
    #         for name in self.supplements_names:
    #             params.remove(name)
    #     else:
    #         supplements = self.supplements
    #     return self.Result(x=params, supplements=supplements)

    def fit_ks(self, values, opt_func=sp_optimize.fmin, x0=None, *args,
               **kwds):
        if x0 is None:
            x0 = (1, ) * len(self.parameter_names)
        return min_ks(self.cdf, x0, values, opt_func, *args, **kwds)

    def fit_fsum(self, values, opt_func=sp_optimize.fmin, x0=None,
                 *args, **kwds):
        if x0 is None:
            x0 = (1, ) * len(self.parameter_names)
        return min_fsum(self.cdf, x0, values, opt_func, *args, **kwds)

    def _constraints(self, *args, **kwds):
        param = kwds[list(kwds.keys())[0]]
        return np.full_like(param, False, dtype=bool)

    def _invalid_x(self, x, *args, **kwds):
        """Returns a mask indicating where x < lower bound or x > upper bound.
        """
        x = np.atleast_1d(x)
        lower_key = ("lc" if "lc" in self.parameter_names else "l"
                     if "l" in self.parameter_names else None)
        lower = kwds.get(lower_key, False)
        b_shape = np.broadcast(x, lower).shape
        mask = np.atleast_1d(np.full(b_shape, False, dtype=bool))
        if isinstance(lower, np.ndarray) or lower:
            if lower is None:
                lower_i = self.parameter_names.index(lower_key)
                lower = np.atleast_1d(args[lower_i])
            if x.size == 1 and isinstance(lower, np.ndarray):
                x = np.full_like(lower, x)
                mask = np.full_like(lower, mask)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mask[x < lower] = True

        upper_key = ("uc" if "uc" in self.parameter_names else "u"
                     if "u" in self.parameter_names else None)
        upper = kwds.get(upper_key, False)
        if isinstance(upper, np.ndarray) or upper:
            if upper is None:
                upper_i = self.parameter_names.index(upper_key)
                upper = np.atleast_1d(args[upper_i])
            if x.size == 1 and isinstance(upper, np.ndarray):
                x = np.full_like(upper, x)
                mask = np.full_like(upper, mask)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mask[x > upper] = True

        return np.squeeze(mask)


class Frozen(object):
    # this idea is stolen from the scipy.stats-module and expanded
    def __init__(self, dist, *params):
        self.dist = dist
        self.params = params
        self.parameter_names = dist.parameter_names
        self.name = dist.name

    def pdf(self, x):
        return self.dist.pdf(x, **self.parameter_dict)

    def cdf(self, x):
        return self.dist.cdf(x, **self.parameter_dict)

    def ppf(self, qq):
        return self.dist.ppf(qq, **self.parameter_dict)

    def sample(self, size):
        return self.dist.sample(size, **self.parameter_dict)

    @property
    def parameter_dict(self):
        return {
            name: value
            for name, value in zip(self.parameter_names, self.params)
        }

    def plot_fit(self, values=None, n_classes=30):
        """Display a combined plot of a histogram, fitted pdf, empirical cdf
        and fitted cdf."""
        if values is None and hasattr(self.dist, "x"):
            values = self.dist.x
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # the histogram of the data
        bins = ax1.hist(values, n_classes, density=True,
                        facecolor='green', alpha=0.75)[1]

        class_middles = 0.5 * (bins[1:] + bins[:-1])
        density = self.pdf(class_middles)
        ax1.plot(class_middles, density, 'r--')

        # the quantile part
        ax2 = ax1.twinx()
        # empirical cdf
        values_sort = np.sort(values)
        ranks_emp = (.5 + np.arange(len(values))) / len(values)
        ax2.plot(values_sort, ranks_emp)
        # theoretical cdf
        x = np.linspace(values.min(), values.max(), 5e2)
        ranks_theory = self.cdf(x)
        ax2.plot(x, ranks_theory, 'r--')
        ax2.grid()
        plt.title(", ".join(" %s: %.3f" % (par_name, par)
                            for par_name, par in self.parameter_dict.items()
                            if par_name != "kernel_data"))
        fig.suptitle(self.dist.name)
        return fig, (ax1, ax2)

    def plot_qq(self, values, *args, **kwds):
        """A qq-plot. Scatters theoretic over empirical ranks."""
        ranks_emp = (stats.stats.rankdata(values) - .5) / len(values)
        ranks_the = self.cdf(values)
        fig = plt.figure()
        plt.scatter(ranks_emp, ranks_the, marker="o", *args, **kwds)
        plt.plot([0, 1], [0, 1])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Empirical ranks")
        plt.ylabel("Fitted ranks")
        plt.grid()
        return fig


class Censored(Dist):
    def __init__(self, distribution, lc=-np.inf, uc=np.inf):
        self.dist = distribution
        self.lc = lc
        self.uc = uc
        # read "lc" as "lower cut", "uc" as "upper cut"
        # mind you: these are different than lower and upper bound in i.e. the
        # beta distribution. we can also truncate the beta distribution and
        # have upper bounds above the upper cut.
        if hasattr(distribution, "parameter_names"):
            self.parameter_names = tuple(
                list(distribution.parameter_names) + ["lc", "uc"])
        self.name = "censored " + distribution.name

    def _pdf(self, x, *args, **kwds):
        x = np.atleast_1d(x)
        args = [np.asarray(arg) for arg in args]
        lc = self.lc if "lc" not in kwds else kwds.pop("lc")
        uc = self.uc if "uc" not in kwds else kwds.pop("uc")
        density = self.dist.pdf(x, *args, **kwds)
        qq = self.dist.cdf(x, *args, **kwds)
        lc_above_ii = np.atleast_1d(x > lc)
        if np.any(lc_above_ii):
            try:
                lc_index = np.argmin(x[lc_above_ii])
                density[lc_index] += qq[lc_index]
            except IndexError:
                for jj, lc_above in enumerate(lc_above_ii):
                    lc_index = np.argmin(x[0, lc_above])
                    density[jj, lc_index] += qq[jj, lc_index]
        uc_below_ii = np.atleast_1d(x < uc)
        if np.any(uc_below_ii):
            try:
                uc_index = np.argmax(x[uc_below_ii])
                density[uc_index] += 1 - qq[uc_index]
            except IndexError:
                for jj, uc_below in enumerate(uc_below_ii):
                    uc_index = np.argmax(x[0, uc_below])
                    density[jj, uc_index] += 1 - qq[jj, uc_index]
        density[(x < lc) | (x > uc)] = 0
        return density

    def _cdf(self, x, *args, **kwds):
        x = np.asarray(x)
        args = [np.asarray(arg) for arg in args]
        lc = self.lc if "lc" not in kwds else kwds.pop("lc")
        uc = self.uc if "uc" not in kwds else kwds.pop("uc")
        qq = np.atleast_1d(self.dist.cdf(x, *args, **kwds))
        below_lc_ii = np.atleast_1d(x <= lc)
        if np.any(below_lc_ii):
            lc_quantile = self.dist.cdf(lc, *args, **kwds)
            if np.isscalar(lc_quantile):
                lc_quantile = np.full_like(x, lc_quantile)
            qq[below_lc_ii] = lc_quantile[below_lc_ii]
        above_uc_ii = np.atleast_1d(x >= uc)
        if np.any(above_uc_ii):
            uc_quantile = self.dist.cdf(uc, *args, **kwds)
            if np.isscalar(uc_quantile):
                uc_quantile = np.full_like(x, uc_quantile)
            qq[above_uc_ii] = 1 - uc_quantile[above_uc_ii]
        qq[qq < 0] = 0
        qq[qq > 1] = 1
        return np.squeeze(qq)

    def _ppf(self, qq, *args, **kwds):
        qq = np.asarray(qq)
        args = [np.asarray(arg) for arg in args]
        lc = self.lc if "lc" not in kwds else kwds["lc"]  # kwds.pop("lc")
        uc = self.uc if "uc" not in kwds else kwds["uc"]  # kwds.pop("uc")
        x = self.dist.ppf(qq, *args, **kwds)
        lower_ii = x < lc
        if np.any(lower_ii):
            x[lower_ii] = (np.full_like(x, lc)
                           if np.isscalar(lc) else lc)[lower_ii]
        upper_ii = x > uc
        if np.any(upper_ii):
            x[upper_ii] = (np.full_like(x, uc)
                           if np.isscalar(uc) else uc)[upper_ii]
        return x

    def _fit(self, x, *args, **kwds):
        # try:
        #     x0 = self.dist._feasible_start
        # except AttributeError:
        #     x0 = self.dist.fit(x, *args, **kwds)
        # using the cdf to fit, because of the insanity inherent in the pdf
        # (i.e. no maximum likelihood)
        # return min_ks(self.cdf, x0, x, lc=self.lc, uc=self.uc, *args)
        # fit on valid x
        valid_x = x[~self._invalid_x(x, *args, **kwds)]
        return self.dist.fit(valid_x, *args, **{
            key: val
            for key, val in list(kwds.items()) if key != "lc" and key != "uc"
        })

    def constraints(self, x, *args, **kwds):
        return self.dist.constraints(x, *args, **kwds)


class Truncated(Dist):
    """Wraps around any other distribution found here to truncate its upper
    and/or lower tail."""

    def __init__(self, distribution, lc=-np.inf, uc=np.inf):
        self.dist = distribution
        self.lc = lc
        self.uc = uc
        # read "lc" as "lower cut", "uc" as "upper cut"
        # mind you: these are different than lower and upper bound in i.e. the
        # beta distribution. we can also truncate the beta distribution and
        # have upper bounds above the upper cut.
        self.parameter_names = tuple(
            list(distribution.parameter_names) + ["lc", "uc"])
        self.name = "truncated " + distribution.name

    def _pdf(self, x, *args, **kwds):
        args = [np.asarray(arg) for arg in args]
        if "uc" in kwds:
            uc = kwds.pop("uc")
            if len(args) > 0:
                del args[-1]
        else:
            uc = self.uc
        if "lc" in kwds:
            lc = kwds.pop("lc")
            if len(args) > 0:
                del args[-1]
        else:
            lc = self.lc
        if len(args) > len(self.dist.parameter_names):
            args = args[:len(self.dist.parameter_names)]
        # still sober
        un_trunc = (self.dist.pdf(x, *args, **kwds) /
                    (self.dist.cdf(uc, *args, **kwds) -
                     self.dist.cdf(lc, *args, **kwds)))
        un_trunc[(x < lc) | (x > uc)] = 1e-9
        # now totally drunkated
        return un_trunc

    def _cdf(self, x, *args, **kwds):
        args = [np.asarray(arg) for arg in args]
        if "uc" in kwds:
            uc = kwds.pop("uc")
            if len(args) > 0:
                del args[-1]
        else:
            uc = self.uc
        if "lc" in kwds:
            lc = kwds.pop("lc")
            if len(args) > 0:
                del args[-1]
        else:
            lc = self.lc
        if len(args) > len(self.dist.parameter_names):
            args = args[:len(self.dist.parameter_names)]
        qq = ((self.dist.cdf(x, *args, **kwds)
               - self.dist.cdf(lc, *args, **kwds))
              /
              (self.dist.cdf(uc, *args, **kwds)
               - self.dist.cdf(lc, *args, **kwds)))
        qq[qq < 0] = 0
        qq[qq > 1] = 1
        return qq

    def _ppf(self, qq, *args, **kwds):
        args = [np.asarray(arg) for arg in args]
        if "uc" in kwds:
            uc = kwds.pop("uc")
            if len(args) > 0:
                del args[-1]
        else:
            uc = self.uc
        if "lc" in kwds:
            lc = kwds.pop("lc")
            if len(args) > 0:
                del args[-1]
        else:
            lc = self.lc
        if len(args) > len(self.dist.parameter_names):
            args = args[:len(self.dist.parameter_names)]
        x = self.dist.ppf(qq * self.dist.cdf(uc, *args, **kwds) +
                          (1 - qq) * self.dist.cdf(lc, *args, **kwds), *args,
                          **kwds)
        lower_ii = x < lc
        if np.sum(lower_ii) > 0:
            x[lower_ii] = lc[lower_ii]
        upper_ii = x > uc
        if np.sum(upper_ii) > 1:
            x[upper_ii] = uc[upper_ii]
        return x

    def _fit(self, x, *args, **kwds):
        try:
            x0 = self.dist._feasible_start
        except AttributeError:
            x0 = self.dist.fit(x, *args)
        # using the cdf to fit, because of the insanity inherent in the pdf
        # (i.e. no maximum likelihood)
        if self.lc is None:
            x0 = tuple(list(x0) + [np.min(x)])
            kwds["lc"] = x0[-1]
        else:
            kwds["lc"] = self.lc
        if self.uc is None:
            x0 = tuple(list(x0) + [np.max(x)])
            kwds["uc"] = x0[-1]
        else:
            kwds["uc"] = self.uc
        solution = list(min_ks(self.cdf, x0, x, *args, **kwds))
        if self.uc is None:
            self.uc = solution.pop(-1)
        if self.lc is None:
            self.lc = solution.pop(-1)
        if "lc" in kwds:
            solution += [kwds["lc"]]
        if "uc" in kwds:
            solution += [kwds["uc"]]
        return solution

    def constraints(self, x, *args, **kwds):
        return self.dist.constraints(x, *args, **kwds)


class Normal(Dist):
    _feasible_start = (0, 2)

    def _pdf(self, x, mu=0, sigma=1):
        x, mu, sigma = np.atleast_1d(x, mu, sigma)
        if NE:
            _pi = np.pi
            dens = ne.evaluate("(1. / (2 * _pi * sigma ** 2) ** .5 *" +
                               "exp(-(x - mu) ** 2 / (2 * sigma ** 2)))")
        else:
            dens = (1. / (2 * np.pi * sigma**2)**.5 *
                    np.exp(-(x - mu)**2 / (2 * sigma**2)))
        return dens

    def _cdf(self, x, mu=0, sigma=1):
        x, mu, sigma = np.atleast_1d(x, mu, sigma)
        qq = .5 * (1 + special.erf((x - mu) / (sigma * 2 ** .5)))
        return qq

    def _ppf(self, qq, mu=0, sigma=1):
        qq, mu, sigma = np.atleast_1d(qq, mu, sigma)
        x = special.ndtri(qq) * sigma + mu
        return x

    def _fit(self, x, mu=None, sigma=None):
        x_noninf = x[np.isfinite(x)]
        if mu is None:
            mu = x_noninf.mean()
        if sigma is None:
            sigma = x_noninf.std()
        return mu, sigma


norm = Normal()

# class SkewNormal(Dist):
#     """This is taken and adapted from
#     http://promanthan.com/randomstuff/skewt-0.0.1.tgz
#     which might end up in scipy.stats"""
#     @staticmethod
#     def _fui(h, i):
#         return (h ** (2 * i)) / ((2 ** i) * gamma(i + 1))
#
#     @staticmethod
#     def _tInt(h, a, jmax, cutPoint):
#         seriesL = np.empty(0)
#         seriesH = np.empty(0)
#         i = np.arange(0, jmax + 1)
#         low = h <= cutPoint
#         hL = h[low]
#         hH = h[np.logical_not(low)]
#         L = hL.size
#         if L > 0:
#             b = SkewNormal._fui(hL[:, np.newaxis], i)
#             cumb = b.cumsum(axis=1)  # transposed compared to R code
#             b1 = np.exp(-0.5 * hL ** 2)[:, np.newaxis] * cumb
#             matr = np.ones((jmax + 1, L)) - b1.transpose()
#             jk = ([1.0, -1.0] * jmax)[0:jmax + 1] / (2 * i + 1)
#             matr = np.inner((jk[:, np.newaxis] * matr).transpose(),
#                             a ** (2 * i + 1.0))
#             seriesL = (np.arctan(a) - matr.flatten(1)) / (2 * np.pi)
#         if hH.size > 0:
#             seriesH = (np.arctan(a) *
#                        np.exp(-0.5 * (hH ** 2.0) * a / np.arctan(a))
#                         * (1 + 0.00868 * (hH ** 4.0) * a ** 4.0) /
#                         (2.0 * np.pi))
#         series = np.empty(h.size)
#         series[low] = seriesL
#         series[np.logical_not(low)] = seriesH
#         return series
#
#     @staticmethod
#     def _tOwen(h, a, jmax=50, cutPoint=6):
#         aa = np.abs(a)
#         ah = np.abs(h)
#         if np.isnan(aa):
#             raise ValueError("a is NaN")
#         if np.isposinf(aa):
#             return 0.5 * norm().cdf(-ah)
#         if aa == 0.0:
#             return np.zeros(h.size)
#         na = np.isnan(h)
#         inf = np.isposinf(ah)
#         ah[np.logical_or(na, inf)] = 0
#         if aa <= 1:
#             owen = SkewNormal._tInt(ah, aa, jmax, cutPoint)
#         else:
#             owen = (0.5 * norm().cdf(ah) + norm().cdf(aa * ah)
#                     * (0.5 - norm().cdf(ah)) -
#                     SkewNormal._tInt(aa * ah, (1.0 / aa), jmax, cutPoint))
#         owen[np.isposinf(owen)] = 0
#         return owen * np.sign(a)
#
#     def _pdf(self, x, zeta, omega, alpha):
#         x, zeta, omega, alpha = np.atleast_1d(x, zeta, omega, alpha)
#         return (1 / (omega * np.pi) *
#                 np.exp(-(x - zeta) ** 2 / (2 * omega ** 2))
#                 * norm.cdf(alpha * (x - zeta) / omega))
#
#     def _cdf(self, x, zeta, omega, alpha):
#         x, zeta, omega, alpha = np.atleast_1d(x, zeta, omega, alpha)
#         return (norm.cdf((x - zeta) / omega) -
#                 2 * SkewNormal._tOwen((x - zeta) / omega, alpha))
#
#     def _ppfInternal(self, qq, shape):
#         maxQ = np.sqrt(chi2.ppf(qq, 1))
#         minQ = -np.sqrt(chi2.ppf(1 - qq, 1))
#         if shape > 1e+5:
#             return maxQ
#         if shape < -1e+5:
#             return minQ
#         nan = np.isnan(qq) | (qq > 1) | (qq < 0)
#         zero = qq == 0
#         one = qq == 1
#         qq[nan | zero | one] = 0.5
#         cum = SkewNormal._cumulants(shape, 4)
#         g1 = cum[2] / cum[1] ** (3 / 2.0)
#         g2 = cum[3] / cum[1] ** 2
#         x = norm().ppf(qq)
#         x = (x + (x ** 2 - 1) * g1 / 6 + x * (x ** 2 - 3) * g2 / 24 -
#              x * (2 * x ** 2 - 5) * g1 ** 2 / 36)
#         x = cum[0] + np.sqrt(cum[1]) * x
#         tol = 1e-8
#         maxErr = 1
#         while maxErr > tol:
#             sn = skewnorm(shape)
#             x1 = x - (sn.cdf(x) - qq) / (sn.pdf(x))
#             x1 = np.minimum(x1, maxQ)
#             x1 = np.maximum(x1, minQ)
#             maxErr = np.amax(np.abs(x1 - x) / (1 + np.abs(x)))
#             x = x1
#         x[nan] = np.NaN
#         x[zero] = -np.Infinity
#         x[one] = np.Infinity
#         return x
#
#     def _ppf(self, qq, shape):
#         if np.all(shape == shape[0]):
#             return self._ppfInternal(qq, shape[0])
#         else:
#             vec = np.vectorize(lambda qq, shape:
#                                 self._ppfInternal(np.array([qq]), shape))
#             return vec(qq, shape)
#
#     def _ppf(self, qq, zeta, omega, alpha):
#         qq, zeta, omega, alpha = \
#             np.atleast_1d(qq, zeta, omega, alpha)
#         return self._ppf(qq, alpha) * omega + zeta


class SkewNormal(Dist):
    _feasible_start = (0, 1, .5)

    def _pdf(self, x, zeta=0, omega=1, alpha=0):
        x, zeta, omega, alpha = np.atleast_1d(x, zeta, omega, alpha)
        t = old_div((x - zeta), omega)
        dens = 2. / omega * norm.pdf(t) * norm.cdf(t * alpha)
        return dens

    def _cdf(self, x, zeta=0, omega=1, alpha=0):
        x, zeta, omega, alpha = np.atleast_1d(x, zeta, omega, alpha)
        t = old_div((x - zeta), omega)
        qq = norm.cdf(t) - 2. * owens_t(t, alpha)
        return qq

    def _ppf(self, qq, zeta=0, omega=1, alpha=0):
        qq, zeta, omega, alpha = \
            np.atleast_1d(qq, zeta, omega, alpha)
        if len(zeta) == 1:
            zeta = np.full_like(qq, zeta)
        if len(omega) == 1:
            omega = np.full_like(qq, omega)
        if len(alpha) == 1:
            alpha = np.full_like(qq, alpha)

        qq0 = norm.cdf(qq) * omega + zeta
        x = np.empty_like(qq)
        for i in range(len(qq0)):
            q_exp = qq[i]

            def error(x):
                q_act = self.cdf(x, zeta[i], omega[i], alpha[i])
                return (q_act - q_exp) ** 2

            x[i] = sp_optimize.minimize(error, qq0[i],
                                        method="Nelder-Mead"
                                        )["x"]
        return x

    def _fit(self, x, skew_max=.9):
        skew = stats.skew(x)
        skew_23 = np.abs(skew)**(old_div(2., 3))
        if not (-skew_max < skew < skew_max):
            delta = np.sqrt(np.pi / 2 * skew_max**(old_div(2., 3)) /
                            (skew_max**(old_div(2., 3)) + (old_div(
                                (4 - np.pi), 2))**(old_div(2., 3))))
            delta = np.copysign(delta, skew)

#             warnings.warn("Sample skew %.2f not in feasible range ~(-1,1)!" %
#                           skew)
        else:
            delta = np.sqrt(np.pi / 2 * skew_23 / (skew_23 + (old_div(
                (4 - np.pi), 2))**(old_div(2., 3))))
        delta = np.copysign(delta, skew)
        alpha = old_div(delta, np.sqrt(1 - delta**2))
        omega = np.sqrt(old_div(np.var(x), (1 - 2 * delta**2 / np.pi)))
        zeta = np.mean(x) - omega * np.sqrt(old_div(2, np.pi)) * delta
        return self.fit_ml(x, x0=(zeta, omega, alpha)).x
        if not (-skew_max < skew < skew_max):
            zeta, omega, alpha = self.fit_ml(
                x, x0=(zeta, omega, old_div(alpha, 2)))
            delta = old_div(alpha, np.sqrt(1 + alpha**2))
        return zeta, omega, alpha
if owens:
    skewnorm = SkewNormal()


class TruncatedNormal(Dist):
    _feasible_start = (0, 1, -1, 1)
    _additional_kwds = {"lc": -1, "uc": 1}
    _bounds = [(-np.inf, np.inf),
               (1e-9, np.inf)]

    def _pdf(self, x, mu=0, sigma=1, lc=-np.inf, uc=np.inf):
        x, mu, sigma, lc, uc = np.atleast_1d(x, mu, sigma, lc, uc)
        un_trunc = np.atleast_1d(
            norm.pdf(x, mu, sigma) /
            (norm.cdf(uc, mu, sigma) - norm.cdf(lc, mu, sigma)))
        un_trunc[(x < lc) | (x > uc)] = 0
        sigma_neg_mask = sigma <= 0
        if np.any(sigma_neg_mask):
            if sigma.size == 1:
                un_trunc += sigma**2
            else:
                un_trunc[sigma_neg_mask] += sigma[sigma_neg_mask]**2
        return un_trunc

    def _cdf(self, x, mu=0, sigma=1, lc=-np.inf, uc=np.inf):
        x, mu, sigma, lc, uc = np.atleast_1d(x, mu, sigma, lc, uc)
        qq = ((norm.cdf(x, mu, sigma) - norm.cdf(lc, mu, sigma)) /
              (norm.cdf(uc, mu, sigma) - norm.cdf(lc, mu, sigma)))
        qq = np.atleast_1d(qq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            qq[qq < 0] = 0
            qq[qq > 1] = 1
        return qq

    def _ppf(self, qq, mu=0, sigma=1, lc=-np.inf, uc=np.inf):
        qq, mu, sigma, lc, uc = \
            np.atleast_1d(qq, mu, sigma, lc, uc)
        x = norm.ppf(qq * norm.cdf(uc, mu, sigma) +
                     (1 - qq) * norm.cdf(lc, mu, sigma),
                     mu, sigma)
        x = np.atleast_1d(x)
        lower_ii = x < lc
        if np.sum(lower_ii) > 0:
            x[lower_ii] = lc[lower_ii]
        upper_ii = x > uc
        if np.sum(upper_ii) > 1:
            x[upper_ii] = uc[upper_ii]
        return x

    def _fit(self, x, lc=-np.inf, uc=np.inf):
        x0 = [np.nanmean(x), np.nanstd(x)]
        kwds = {}
        if lc is None:  # or np.isneginf(lc):
            x0 += [np.nanmin(x)]
        else:
            kwds["lc"] = lc
        if uc is None:  # or np.isinf(uc):
            x0 += [np.nanmax(x)]
        else:
            kwds["uc"] = uc
        return self.fit_ml(x, x0=x0, method="Nelder-Mead", **kwds).x

    def _constraints(self, x, mu, sigma, lc=-np.inf, uc=np.inf, **kwds):
        mask = ((x > uc) |
                (x < lc) |
                (sigma <= 0))
        return mask
truncnorm = TruncatedNormal()


class LogNormal(Dist):
    _feasible_start = (2., .5)

    def _pdf(self, x, mu, sigma):
        return norm.pdf(np.log(x), mu, sigma) / x

    def _cdf(self, x, mu, sigma):
        return norm.cdf(np.log(x), mu, sigma)

    def _ppf(self, qq, mu, sigma):
        return np.exp(norm.ppf(qq, mu, sigma))

    def _fit(self, x):
        x_noninf = x[(x > 0) & np.isfinite(x)]
        mu = np.mean(np.log(x_noninf))
        sigma = np.std(np.log(x_noninf))
        return mu, sigma

    def _constraints(self, x, mu, sigma):
        mask = ((x <= 0) |
                (sigma <= 0))
        return mask
lognormal = LogNormal()


class JohnsonSU(Dist):
    _feasible_start = (1, 1, 0, 1)

    def _pdf(self, x, a, b, loc, scale):
        x, a, b, loc, scale = np.atleast_1d(x, a, b, loc, scale)
        x = (x - loc) / scale
        x2 = x * x
        trm = norm.pdf(a + b * np.log(x + np.sqrt(x2 + 1)))
        dens = b * 1.0 / np.sqrt(x2 + 1.0) * trm / scale
        return dens

    def _cdf(self, x, a, b, loc, scale):
        x = np.atleast_1d(x - loc) / scale
        qq = np.atleast_1d(norm.cdf(a + b * np.log(x + np.sqrt(x * x + 1))))
        qq[np.isneginf(x)] = 0
        return qq

    def _ppf(self, qq, a, b, loc, scale):
        qq, a, b, loc, scale = np.atleast_1d(qq, a, b, loc, scale)
        z = np.sinh((norm.ppf(qq) - a) / b)
        x = z * scale + loc
        return x

    def _fit(self, x):
        x_noinf = x[np.isfinite(x)]
        loc, scale = x_noinf.mean(), x_noinf.std()
        return self.fit_fsum(x_noinf, x0=(1, 1, loc, scale))

    def _constraints(self, x, a, b, loc, scale):
        return b <= 0
johnsonsu = JohnsonSU()


class Cauchy(Dist):
    _feasible_start = (0, 1)

    def _pdf(self, x, x0, gamma):
        return 1 / (np.pi * gamma * (1 + ((x - x0) / gamma)**2))

    def _cdf(self, x, x0, gamma):
        return (1 / np.pi) * np.arctan2(x - x0, gamma) + .5

    def _ppf(self, qq, x0, gamma):
        return x0 + gamma * np.tan(np.pi * (qq - .5))

    def _fit(self, x):
        x0 = np.median(stats.trimboth(x, .38))
        inner_quart = stats.trimboth(x, .25)
        inner_range = inner_quart.max() - inner_quart.min()
        return self.fit_ml(x, x0=(x0, .5 * inner_range)).x

    def _constraints(self, x, x0, gamma):
        return gamma < 0
cauchy = Cauchy()


class StudentT(Dist):
    _feasible_start = (0, 1)

    def _pdf(self, x, mu=0, df=1):
        x, mu, df = np.atleast_1d(x, mu, df)
        Px = (np.exp(gammaln((df + 1) / 2.) - gammaln(df / 2.)) /
              np.sqrt(df * np.pi) * (1 + 
                  (x - mu)**2 / df)**(-(df + 1) / 2.))
        return Px

    def _cdf(self, x, mu=0, df=1):
        qq = stdtr(df, np.atleast_1d(x) - mu)
        return qq

    def _ppf(self, qq, mu=0, df=1):
        qq = np.atleast_1d(qq)
        x = stdtrit(df, qq) + mu
        x[qq == 0] = -np.inf
        x[qq == 1] = np.inf
        return x

    def _fit(self, x):
        x_noninf = x[np.isfinite(x)]
        sigma = np.var(x_noninf)
        df = 2 * sigma / (sigma - 1)
        if df > 6:
            df = 6
        return self.fit_ml(
            x_noninf,
            x0=(np.mean(x_noninf), df if df > 0 else .1),
            method="Nelder-Mead").x

    def _constraints(self, x, mu, df):
        mask = ((df <= 0) |
                (df > 8))
        return mask
student_t = StudentT()


class NoncentralT(Dist):
    _feasible_start = (2, 3, 0)

    # stolen from scipy.stats.distributions.nct
    def _pdf(self, x, df, nc, mu=0):
        x, df, nc = (np.atleast_1d(var).astype(float) for var in (x, df, nc))
        pdf_x = lambda x: ((df / x) *
                           (self.cdf(x * (1 + 2 / df) ** .5, df + 2, nc) -
                            self.cdf(x, df, nc)))
        dens = np.where(
            np.abs(x - mu) <= .5, self._pdf_old(x, df, nc, mu), pdf_x(x - mu))
        dens[np.isinf(x)] = 0
        return dens

    def _pdf_old(self, x, df, nc, mu=0):
        n = df * 1.0
        nc = nc * 1.0
        x2 = (x - mu)**2
        ncx2 = nc * nc * x2
        fac1 = n + x2
        trm1 = (n / 2. * np.log(n) + gammaln(n + 1) -
                (n * np.log(2) + nc * nc / 2. +
                 (n / 2.) * np.log(fac1) + gammaln(n / 2.)))
        Px = np.exp(trm1)
        valF = ncx2 / (2 * fac1)
        trm1 = (sqrt2 * nc *
                (x - mu) * hyp1f1(n / 2 + 1, 1.5, valF) /
                (np.asarray(fac1 * gamma_func((n + 1) / 2))))
        trm2 = (hyp1f1((n + 1) / 2, 0.5, valF) / (np.asarray(
                np.sqrt(fac1) * gamma_func(n / 2 + 1))))
        Px *= trm1 + trm2
        return Px

    def _cdf(self, x, df, nc, mu=0):
        x, df, nc, mu = (np.atleast_1d(var).astype(float)
                         for var in (x, df, nc, mu))
        qq = nctdtr(df, nc, x - mu)
        return qq

    def _ppf(self, qq, df, nc, mu=0):
        qq, df, nc = (np.atleast_1d(var).astype(float)
                      for var in (qq, df, nc))
        x = nctdtrit(df, nc, qq) + mu
        x[qq == 0] = -np.inf
        x[qq == 1] = np.inf
        return x

    def _fit(self, x):
        x_noninf = x[np.isfinite(x)]
        mode = stats.mode(np.round(x_noninf, 1))[0][0]
        x0 = (1, 2 * stats.skew(x_noninf), mode)
        df, nc, mu = self.fit_ml(x_noninf, x0=x0, method="Nelder-Mead").x
        #        if df >= 30:
        #            # then we are normal anyway
        #            df = 30
        return df, nc, mu

    def _constraints(self, x, df, nc, mu):
        mask = ((df < .1) |
                (df > 8))
        return mask


noncentral_t = NoncentralT()

# TODO: implement ppf
# class LogitNormal(Dist):
#    def logit(self, x):
#        return np.log(x / (1 - x))
#
#    def _pdf(self, x, mu, sigma, l=0, u=1):
#        x_normed = (x - l) / (u - l)
#        logit = self.logit(x_normed)
#        return norm_pdf(logit, mu, sigma) / (x_normed * (1 - x_normed))
#
#    def _cdf(self, x, mu, sigma, l=0, u=1):
#        x_normed = (x - l) / (u - l)
#        logit = self.logit(x_normed)
#        return norm_cdf(logit, mu, sigma)
#
#    def _ppf(self, qq, mu, sigma, l=0, u=1):
#        raise NotImplementedError
#
#    def _fit(self, x, l=None, u=None):
#        l = x.min() - 1e-9 if l is None else l
#        u = x.max() + 1e-9 if u is None else u
#        x_normed = (x - l) / (u - l)
#        logit = self.logit(x_normed)
#        mu, sigma = norm_fit(logit)
#        return mu, sigma, l, u
#
#    def _constraints(self, x, mu, sigma, l=0, u=1):
#        x, mu, sigma, l, u = (np.asarray(var) for var in (x, mu, sigma, l, u))
#        if np.any(x < l):
#            return False
#        if np.any(x > u):
#            return False
#        return True
# logitnormal = LogitNormal()

# class Rayleigh(Dist):
#    def _pdf(self, x, sigma):
#        return (x / sigma ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2))
#
#    def _cdf(self, x, sigma):
#        return 1 - np.exp(-x ** 2 / (2 * sigma ** 2))
#
#    def _fit(self, x):
#        return ((.5 * np.average(x ** 2)) ** .5,)
#
#    @abstractmethod
#    def _ppf(self):
#        raise NotImplementedError
#
#    def _constraints(self, x, sigma):
#        if np.any(sigma <= 0):
#            return False
#        return True
# rayleigh = Rayleigh()

# TODO: implement ppf
# class RayleighLU(Dist):
#    def _pdf(self, x, sigma, l, u):
#        x_normed = (x - l) / (u - l)
#        return (x_normed / sigma ** 2) * \
#            np.exp(-x_normed ** 2 / (2 * sigma ** 2))
#
#    def _cdf(self, x, sigma, l, u):
#        x_normed = (x - l) / (u - l)
#        return 1 - np.exp(-x_normed ** 2 / (2 * sigma ** 2))
#
#    def _fit(self, x, l=None, u=None):
#        l = x.min() if l is None else l
#        u = x.max() if u is None else u
#        x_normed = (x - l) / (u - l)
#        return (.5 * np.average(x_normed ** 2)) ** .5, l, u
#
#    def _ppf(self):
#        raise NotImplementedError
#
#    def _constraints(self, x, sigma, l, u):
#        if np.any(sigma <= 0):
#            return False
#        if np.any(l >= u):
#            return False
#        return True
# rayleighlu = RayleighLU()

# TODO: fit does not work well
# class LogNormalLU(Dist):
#    def _pdf(self, x, mu, sigma, l, u):
#        x_normed = (x - l) / (u - l)
#        return norm_pdf(np.log(x_normed + 1e-9), mu, sigma) / (u - l)
#
#    def _cdf(self, x, mu, sigma, l, u):
#        x_normed = (x - l) / (u - l)
#        return norm_cdf(np.log(x_normed + 1e-9), mu, sigma)
#
#    def _ppf(self, qq, mu, sigma, l, u):
#        return np.exp(norm_ppf(qq, mu, sigma)) * (u - l) + l
#
#    def _fit(self, x, l=None, u=None):
#        x_noninf = x[np.isfinite(x)]
#        l = x_noninf.min() if l is None else l
#        u = x_noninf.max() if u is None else u
#        x_normed = (x_noninf - l) / (u - l)
#        mu, sigma = norm_fit(np.log(x_normed + 1e-9))
#        return self.fit_ml(x_noninf, x0=(mu, sigma, l, u))
#
#    def _constraints(self, x, mu, sigma, l, u):
#        if np.any(l >= u):
#            return False
#        return True
# lognormallu = LogNormalLU()
# lognormallu._feasible_start = (0, 1, -1, 1)

# class WeibullLU(Dist):
#    def _pdf(self, x, alpha, beta, l=0, u=1):
#        x_normed = (x - l) / (u - l)
#        # alpha and beta should be positive
#        densities = alpha * beta * x_normed ** (beta - 1) * \
#            np.exp(-alpha * x_normed ** beta)
#        return densities / (u - l)
#
#    def _cdf(self, x, alpha, beta, l=0, u=1):
#        x_normed = (x - l) / (u - l)
#        return 1 - np.exp(-alpha * x_normed ** beta)
#
#    def _ppf(self, qq, alpha, beta, l=0, u=1):
#        x = (-np.log(1 - qq) / alpha) ** (1 / beta)
#        return x * (u - l) + l
#
#    def _fit(self, values, beta_start=10, *args, **kwds):
#        """Implements a semi-analytical method of moments. beta is found by
#        minimizing errors. After that, alpha can be estimated with the help of
#        beta.
#        See
#        http://interstat.statjournals.net/YEAR/2000/articles/0010001.norm_pdf
#        """
#        values_noninf = values[np.isfinite(values)]
#        l = values_noninf.min()
#        u = values_noninf.max()
#        values_noninf = (values_noninf - l) / (u - l)
#        # we need the coefficient of variation now and the mean later. so do
#        # not use stats.variation in order to not calculate the mean twice.
#        mean = np.mean(values_noninf)
#        cv = np.std(values_noninf) / mean
#
#        def cv_error(beta):
#            """Squared error between the theoretical and empirical coefficient
#            of variation."""
#            gamma1 = special.gamma(1 + 2 / beta)
#            gamma2 = special.gamma(1 + 1 / beta)
#            return (cv - (gamma1 - gamma2 ** 2) ** .5 / gamma2) ** 2
#        beta = sp_optimize.fmin(cv_error, [beta_start], disp=False)[0]
#        alpha = (mean / special.gamma(1 / beta + 1)) ** (-beta)
#        return alpha, beta, l, u
#
#    def _constraints(self, x, alpha, beta, l, u):
#        x, alpha, beta = (np.asarray(var) for var in (x, alpha, beta))
#        if np.any(alpha <= 0):
#            return False
#        if np.any(beta <= 0):
#            return False
#        if np.any(x < l):
#            return False
#        if np.any(x > u):
#            return False
#        return True
# weibulllu = WeibullLU()
# weibulllu._feasible_start = (1, 1, -1, 1)


class Weibull(Dist):
    _feasible_start = (.5, 2.0)

    def _pdf(self, x, alpha, beta):
        # alpha and beta should be positive
        densities = alpha * beta * x ** (beta - 1) * \
            np.exp(-alpha * x ** beta)
        return densities

    def _cdf(self, x, alpha, beta):
        return 1 - np.exp(-alpha * x**beta)

    def _ppf(self, qq, alpha, beta):
        return (-np.log(1 - qq) / alpha)**(1. / beta)

    def _fit(self, values, beta_start=10, *args, **kwds):
        """Implements a semi-analytical method of moments. beta is found by
        minimizing errors. After that, alpha can be estimated with the help of
        beta.
        See
        http://interstat.statjournals.net/YEAR/2000/articles/0010001.norm_pdf
        """
        # we need the coefficient of variation now and the mean later. so do
        # not use stats.variation in order to not calculate the mean twice.
        mean = np.nanmean(values)
        cv = np.nanstd(values) / mean

        def cv_error(beta):
            """Squared error between the theoretical and empirical coefficient
            of variation."""
            gamma1 = special.gamma(1 + 2 / beta)
            gamma2 = special.gamma(1 + 1 / beta)
            return (cv - (gamma1 - gamma2**2)**.5 / gamma2)**2

        beta = sp_optimize.fminbound(cv_error, -1, 1e6, disp=False)
        alpha = (mean / special.gamma(1 / beta + 1))**(-beta)
        return alpha, beta

    def _constraints(self, x, alpha, beta):
        mask = ((x < 0) |
                (alpha <= 0) |
                (beta <= 0))
        return mask


weibull = Weibull()


class Kumaraswamy(Dist):
    """Resembles the Beta distribution, but does not need a transcendental
    function."""
    _feasible_start = (1, 1, 0, 1)

    def _pdf(self, x, a, b, l=0, u=1):
        x = np.atleast_1d(x).astype(float)
        if NE:
            x = ne.evaluate("(x - l) / (u - l)")
            dens = ne.evaluate("a * b * x ** (a - 1) * (1 - x ** a) ** " +
                               "(b - 1) / (u - l)")
        else:
            x = (x - l) / (u - l)
            dens = a * b * x**(a - 1) * (1 - x**a)**(b - 1) / (u - l)
        return dens

    def _cdf(self, x, a, b, l=0, u=1):
        x = np.atleast_1d(x).astype(float)
        if np.any((x < l) | (x > u)):
            warnings.warn("Some values below lower or above upper bounds.")
            x = np.where(x > u, u, x)
            x = np.where(x < l, l, x)
        return 1 - (1 - ((x - l) / (u - l)) ** a) ** b

    def _ppf(self, qq, a, b, l=0, u=1):
        qq, a, b, l, u = np.atleast_1d(qq, a, b, l, u)
        if NE:
            x = ne.evaluate("((1 - (1 - qq) ** (1 / b)) ** " +
                            "(1 / a)) * (u - l) + l")
        else:
            x = ((1 - (1 - qq) ** (1. / b)) ** (1. / a)) * (u - l) + l
        return x

    def _fit(self, x, l=None, u=None):
        return beta.fit(x, l, u)

    def _constraints(self, x, a, b, l=0, u=1):
        mask = ((a < 0) |
                (b < 0) |
                (x < l) |
                (x > u))
        return mask

kumaraswamy = Kumaraswamy()


class Beta(Dist):
    """Beta distribution on the interval [l, u]."""
    _feasible_start = (1, 1, 0, 1)

    def _pdf(self, x, alpha, beta, l=0, u=1):
        x, alpha, beta, l, u = list(map(np.asarray, (x, alpha, beta, l, u)))
        # putting the values into [0, 1] according to [l, u]
        # we avoid the value of exactly 0 or 1 to not get numerical problems
        #        x_normed = (x - l) / (u - l)
        #        x = np.copy(x_normed)
        #        ll, uu = sys.float_info.min, 1 - sys.float_info.min
        #        x_normed = np.where(x_normed >= uu, uu, x_normed)
        #        x_normed = np.where(x_normed <= ll, ll, x_normed)
        if NE:
            x = ne.evaluate("(x - l) / (u - l)")
            return 1 / special.beta(alpha, beta) * \
                ne.evaluate("x ** (alpha - 1) * (1 - x) ** (beta - 1) / " +
                            "(u - l)")
        else:
            x = (u - l) / (u - l)
            return (1 / special.beta(alpha, beta)
                    * x ** (alpha - 1) * (1 - x) ** (beta - 1) / (u - l))

    def _cdf(self, x, alpha, beta, l=0, u=1):
        if np.any((x < l) | (x > u)):
            warnings.warn("Some values below lower or above upper bounds.")
        # note: the betainc function below really delivers the regularized
        # incomplete beta function
        return special.betainc(alpha, beta, (x - l) / (u - l))

    def _ppf(self, qq, alpha, beta, l=0, u=1):
        return (u - l) * special.betaincinv(alpha, beta, qq) + l

    def _fit(self, x, l=None, u=None):
        """stolen from
        http://en.wikipedia.org/wiki/Beta_distribution#Parameter_estimation."""
        # remember whether l and u where given. if they were, we do
        # not return them
        if l is None:
            l = x.min()
            return_l = True
        else:
            return_l = False
        if u is None:
            u = x.max()
            return_u = True
        else:
            return_u = False

        # have to make sure that the parameters all have the same lengths
        try:
            ll = np.atleast_1d(np.empty_like(u))
            ll[:] = l
            l = ll
        except (TypeError, ValueError):
            pass
        try:
            uu = np.empty_like(l)
            uu[:] = u
            u = uu
        except (TypeError, ValueError):
            pass

        xmean = (x.mean() - l) / (u - l)
        xvar = x.var() / (u - l) ** 2
        alpha = xmean * (xmean * (1 - xmean) / xvar - 1)
        beta = (1 - xmean) * alpha / xmean

        return_list = [abs(alpha), abs(beta)]
        if return_l:
            return_list += [l]
        if return_u:
            return_list += [u]
        return np.squeeze(return_list)

#        return self.fit_ml(x, x0=(alpha, beta, l, u))

    def _constraints(self, x, alpha, beta, l=0, u=1):
        mask = ((alpha <= 0) |
                (beta <= 0) |
                (x < l) |
                (x >u))
        return mask
beta = Beta()


class Gamma(Dist):
    """Gamma distribution."""
    _feasible_start = (2, 2)

    def _pdf(self, x, k, theta):
        x = np.atleast_1d(x)
        dens = (x**(k - 1) * np.exp(-x / float(theta)) /
                (special.gamma(k) * theta**k))
        return dens

    def _cdf(self, x, k, theta):
        x = np.atleast_1d(x)
        qq = gammainc(k, x / theta) / gamma_func(k)
        # qq[np.isinf(x)] = 1
        return qq

    def _ppf(self, qq, k, theta):
        qq = np.atleast_1d(qq)
        x = theta * gammaincinv(k, qq * gamma_func(k))
        # x[qq == 1] = np.inf
        return x

    def _fit(self, x, rel_change=1e-6):
        """A maximum-likelihood estimator. See
        http://en.wikipedia.org/wiki/Gamma_distribution#Parameter_estimation
        """

        def psi(k):
            if k >= 8:
                return (np.log(k) -
                        (1. + (1. - (.1 - 1. / (21. * k ** 2)) / k ** 2)
                         / (6. * k)) / (2. * k))
            else:
                return psi(k + 1) - 1. / k

        def psi_(k):
            if k >= 8:
                return ((1. + (1. + (1. - (1. / 5. - 1. / (7. * k ** 2)) /
                                     k ** 2) / (3. * k)) / (2. * k)) / k)
            else:
                return psi_(k + 1) + 1. / k**2

        xmean = x[np.isfinite(x)].mean()
        # we get problems here if there are 0's in x (log(x) = -inf, which
        # happens for wind)
        s = np.log(xmean) - np.mean(np.log(x[np.isfinite(x) & (x > 0)]))
        k_old = (3 - s + ((s - 3)**2 + 24 * s)**.5) / (12 * s)
        k_rec = lambda k: k - (np.log(k) - psi(k) - s) / (1. / k - psi_(k))
        k_new = k_rec(k_old)
        while abs(k_old - k_new) / k_old > rel_change:
            k_old = k_new
            k_new = k_rec(k_old)
        theta = xmean / k_new
        return self.fit_fsum(x, x0=(k_new, theta))

    def _constraints(self, x, k, theta):
        mask = ((k <= 0) |
                (theta <= 0) |
                (x < 0))
        return mask


gamma = Gamma()


class Gamma1(Dist):
    _feasible_start = (1, .5, 2)

    def _pdf(self, x, a, loc, scale):
        x, a, loc, scale = np.atleast_1d(x, a, loc, scale)
        z = (x - loc) / scale
        dens = (np.exp((a - 1) * np.log(z) - z - gammaln(a))) / scale

        return dens

    def _cdf(self, x, a, loc, scale):
        x, a, loc, scale = np.atleast_1d(x, a, loc, scale)
        z = (x - loc) / scale
        qq = gammainc(a, z)
        qq[np.isinf(x)] = 1
        return qq

    def _ppf(self, qq, a, loc, scale):
        qq, a, loc, scale = np.atleast_1d(qq, a, loc, scale)
        x = gammaincinv(a, qq) * scale + loc
        x[qq == 1] = np.inf
        return x

    def _fit(self, x):
        return stats.gamma.fit(x[np.isfinite(x)])
        # this is stolen from sp.stats.gamma-gen and not understood

    #        x_noninf = x[np.isfinite(x)]
    #        a = 4 / stats.skew(x_noninf) ** 2
    #        muhat = np.array(x_noninf).mean()
    #        mu2hat = np.array(x_noninf).var()
    #        Shat = (mu2hat / a) ** .5
    #        Lhat = muhat - Shat * a
    #        return self.fit_ml(x_noninf, x0=(a, Lhat, Shat))


gamma1 = Gamma1()


class Expon(Dist):
    _feasible_start = (1e-3, 1. / 5)

    def _pdf(self, x, x0, lambd):
        return np.where(x > x0,
                        lambd * np.exp(-lambd * (x - x0)),
                        0)

    def _cdf(self, x, x0, lambd):
        return np.where(x > x0,
                        1 - np.exp(-lambd * (x - x0)),
                        0)

    def _ppf(self, qq, x0, lambd):
        return -np.log(1 - qq) / lambd + x0

    def _fit(self, x):
        x_noninf = x[np.isfinite(x)]
        x0 = np.min(x_noninf)
        lambd = 1 / np.mean(x_noninf)
        return x0, lambd

    def _constraints(self, x, x0, lambd):
        return x <= x0


expon = Expon()


class ExponTwo(Dist):
    _feasible_start = (0, .5, 1, 1)

    def _pdf(self, x, x0, q0, lambda1, lambda2):
        x = np.atleast_1d(x)
        dens = np.where(x <= x0, q0 * expon.pdf(x0 - x, 0, lambda1),
                        (1 - q0) * expon.pdf(x, x0, lambda2))
        return dens

    def _cdf(self, x, x0, q0, lambda1, lambda2):
        x = np.atleast_1d(x)
        qq = np.where(x <= x0, q0 * (1 - expon.cdf(x0 - x, 0, lambda1)),
                     q0 + (1 - q0) * expon.cdf(x, x0, lambda2))
        return qq

    def _ppf(self, qq, x0, q0, lambda1, lambda2):
        qq = np.atleast_1d(qq)
        x = np.where(
            qq <= q0, x0 + np.log(qq / q0) / lambda1,
            x0 - np.log(1 - (qq - q0) / (1 - q0)) / lambda2)
        return x

    def _fit(self, x):
        x_noninf = x[np.isfinite(x)]
        x0 = stats.mode(np.round(x_noninf, 1))[0][0]
        lambda1 = -1 / np.mean(x_noninf[x_noninf < x0] - x0)
        lambda2 = 1 / np.mean(x_noninf[x_noninf > x0] - x0)
        q0 = (np.argmin(np.abs(np.sort(x_noninf) - x0)) + .5) / len(x)
        return self.fit_ml(
            x_noninf, x0=(x0, q0, lambda1, lambda2), method="Nelder-Mead").x

    def _constraints(self, x, x0, q0, lambda1, lambda2):
        mask = ((lambda1 <= 0) |
                (lambda2 <= 0))
        return mask


expon_two = ExponTwo()


class NoncentralLaplace(Dist):
    _feasible_start = (1, .5, .5)

    def _pdf(self, x, x0, q0, lambd):
        lambda2 = lambd * q0 / (1 - q0)
        x = np.atleast_1d(x)
        dens = np.where(
            x <= x0,
            q0 * lambd * np.exp(-lambd * (x0 - x)),
            (1 - q0) * lambda2 * np.exp(-lambda2 * (x - x0)), )
        dens[np.isinf(x)] = 0
        return dens

    def _cdf(self, x, x0, q0, lambd):
        lambda2 = lambd * q0 / (1 - q0)
        x = np.atleast_1d(x)
        qq = np.where(x <= x0, q0 * (1 - expon.cdf(x0 - x, 0, lambd)),
                     q0 + (1 - q0) * expon.cdf(x, x0, lambda2))
        return qq

    def _ppf(self, qq, x0, q0, lambd):
        lambda2 = lambd * q0 / (1 - q0)
        qq = np.atleast_1d(qq)
        x = np.where(
            qq <= q0, x0 + np.log(qq / q0) / lambd,
            x0 - np.log(1 - (qq - q0) / (1 - q0)) / lambda2)
        return x

    def _fit(self, x):
        x0 = stats.mode(np.round(x, 1))[0][0]
        q0 = (np.argmin(np.abs(np.sort(x) - x0)) + .5) / len(x)
        lambd = (1 - 2 * q0) / (q0 * np.mean(x[np.isfinite(x)]))
        if lambd < 0:
            # this actually means that x0 was not right, so provide a feasible
            # lambd and let the ml estimation figure the rest out
            lambd = -1. / np.mean(x[(x < x0) & np.isfinite(x)] - x0)
        return self.fit_ks(x, x0=(x0, q0, lambd))

    def _constraints(self, x, x0, q0, lambd):
        mask = ((lambd <= 0) |
                (q0 < 0) |
                (q0 > 1))
        return mask


noncentral_laplace = NoncentralLaplace()


class MDFt(object):
    """Multiple degrees of freedom t distribution.

    The marginals have different degrees of freedom, therefore this is
    not a "true" Multivariate t distribution.
    See Serban et al 2007.
    """

    @staticmethod
    def _marginal_pdf(x, sigma, df):
        x = np.atleast_1d(np.asarray(x))
        dens = (gamma_func((df + 1) / 2.) /
                (np.sqrt(df * np.pi) * gamma_func(df / 2.)) *
                (1 + x**2 / df)**(-(df + 1) / 2.))
        return dens

    @staticmethod
    def loglikelihood(data, sigma, df):
        if np.any(~np.isfinite(sigma)):
            return np.inf
        sigma_inv_sqrt = np.linalg.inv(linalg.sqrtm(sigma))
        yt = np.array([np.squeeze(np.asarray(sigma_inv_sqrt[i] @ data))/
                       np.sqrt((dfi - 2) / dfi)
                       for i, dfi in enumerate(df)
                       ])
        marginal_pdfs = [MDFt._marginal_pdf(yt, sigma, dfi) for dfi in df]
        llh = (
            np.sum(np.log(np.sqrt(df - 2) / df)) +
            np.sum(list(map(np.log, marginal_pdfs)))
            # is that needed? it should not change the location of
            # the maximum?!
            - .5 * np.log(linalg.det(sigma)))
        return -llh

    @staticmethod
    def sample(size, sigma, df):
        K = sigma.shape[0]
        if isinstance(df, float):
            df = np.array(K * [df])
        yt = np.array([student_t.sample(size, df=dfi) for dfi in df])
        zt = np.sqrt((df - 2.) / df)[:, None] * yt
        return np.dot(linalg.sqrtm(sigma), zt)

    @staticmethod
    def fit(data):
        K = data.shape[0]
        data = np.asmatrix(data)

        def unlikelihood(params):
            sigma = fill_lower(params[:-K])
            df = params[-K:]
            return -MDFt.loglikelihood(data, sigma, df)

        sigma = np.cov(data)
        sigma_upper = sigma[np.triu_indices_from(sigma)]
        df = np.array(
            [student_t.fit(np.asarray(values))[1] for values in data])
        # df[df < 5] = 5.
        from scipy import optimize
        x0 = np.concatenate((sigma_upper, df))
        bounds = [(0, None) if i == j else (None, None)
                  for i in range(K) for j in range(i + 1)]
        bounds += K * [(5, None)]
        result = optimize.minimize(
            unlikelihood, x0=x0, bounds=bounds, options=dict(disp=True))
        sigma = fill_lower(result.x[:-K])
        df = result.x[-K:]
        return sigma, df


def fill_lower(sequence):
    K = int(np.ceil(np.sqrt(len(sequence))))
    arr = np.empty((K, K))
    arr[np.triu_indices_from(arr)] = sequence
    upper_i = np.triu_indices_from(arr, k=1)
    lower_i = np.tril_indices_from(arr, k=-1)
    arr[lower_i] = arr[upper_i]
    return arr


class _Rain(Dist):
    def _mask_kwds(self, mask, kwds):
        return {
            name: (vals[mask]
                   if (hasattr(vals, "shape") and
                       vals.shape == mask.shape)
                   else vals)
            for name, vals in kwds.items()
        }

    def _pdf(self, meta, *args, **kwds):
        pass

    def _cdf(self, meta, *args, **kwds):
        pass

    def _ppf(self, meta, *args, **kwds):
        pass


class _KDE(object):
    # these are not part of the solution and should not be tested on
    _cache_names = "kernel_data", "x_eval", "q_kde_eval",  # "kernel_width"

    def fit_kde(self, x):
        return kde.optimal_kernel_width(x)

    def _kde_integral(self, kernel_width, kernel_data, f_thresh, upper_eval,
                    recalc=True):
        if kernel_width is None:
            kernel_width = kde.silvermans_rule(kernel_data)
        # lower_eval = max(f_thresh - 1e-6, 1e-9)
        lower_eval = max(.95 * f_thresh, 1e-9)
        if upper_eval is None or np.any(upper_eval < kernel_data):
            upper_eval = 2 * kernel_data.max() + 10 * kernel_width
        x_eval_log = np.log(np.linspace(lower_eval, upper_eval, 500))
        # assert np.all(np.isfinite(x_eval_log))
        assert np.all(kernel_data > 0)
        dens_kde_eval = kde.kernel_density(
            kernel_width, np.log(kernel_data), eval_points=x_eval_log,
            recalc=recalc)

        x_eval = np.exp(x_eval_log)
        dens_kde_eval /= x_eval
        q_kde_eval = cumtrapz(y=dens_kde_eval, x=x_eval, initial=0)
        # this is ugly, but I cannot find a better solution (now)
        q_kde_eval /= q_kde_eval[-1]
        # fill only the quantile range [q_thresh, 1]
        q_kde_eval = self.q_thresh + (1. - self.q_thresh) * q_kde_eval
        q_kde_eval = np.concatenate((q_kde_eval, [1.]))
        x_eval_log = np.concatenate((x_eval_log,
                                     [np.log(upper_eval * 2)]))
        return q_kde_eval, x_eval_log

    def _kde_cdf(self, xx, q_kde_eval=None, x_eval=None):
        xx = np.atleast_1d(xx)
        qq_kde = np.empty_like(xx)
        for i, x in enumerate(xx):
            x_eval_ = (x_eval
                       if x_eval.ndim == 1 else
                       x_eval[i])
            q_kde_eval_ = (q_kde_eval
                           if q_kde_eval.ndim == 1 else
                           q_kde_eval[i])
            q_kde_interp = interpolate.interp1d(x_eval_,
                                                q_kde_eval_,
                                                kind="linear",
                                                assume_sorted=True)
            qq_kde[i] = q_kde_interp(x)
        return qq_kde

    def _kde_ppf(self, qq, q_kde_eval=None, x_eval=None):
        qq = np.atleast_1d(qq)
        # assert np.all(qq >= self.q_thresh)
        xx_kde = np.empty_like(qq)
        for i, q in enumerate(qq):
            x_eval_ = (x_eval
                       if x_eval.ndim == 1 else
                       x_eval[i])
            q_kde_eval_ = (q_kde_eval
                           if q_kde_eval.ndim == 1 else
                           q_kde_eval[i])
            x_kde_interp = interpolate.interp1d(q_kde_eval_,
                                                x_eval_,
                                                kind="linear",
                                                assume_sorted=True)
            xx_kde[i] = x_kde_interp(q)
        return xx_kde


class RainMix(_KDE, _Rain):
    supplements_names = ("q_thresh",
                         "f_thresh",
                         "kernel_data",
                         "q_kde_eval",
                         "x_eval")

    def __init__(self, distribution, threshold=.0015,
               q_thresh_lower=.6, q_thresh_upper=.95):
        """Mixed Rain distribution with lower threshold and KDE above
        q_threshold.

        Requires self-generated sample data to be initialized because
        of the KDE.

        Parameter
        ---------
        distribution : Dist
        threshold : float, optional
            Threshold rain intensity
        q_threshold : float, optional
            Quantile above which to estimate the distribution via KDE.
            Not the quantile of the full distribution, but of the
            'wet' values.
        """
        self.dist = distribution
        self.thresh = threshold
        self.q_thresh_lower = q_thresh_lower
        self.q_thresh_upper = q_thresh_upper
        self.debug = False

        # this enables testing
        sample_data = self._gen_sample_data()
        self.sample_data = sample_data
        self._feasible_start = self.fit(sample_data)
        # parameter_names exists because of the DistMeta class
        self.parameter_names = (self.parameter_names +
                                distribution.parameter_names)
        # n_pars is set in class construction time with the Dist metaclass
        self.n_pars += self.dist.n_pars
        self.name = "rainmix " + distribution.name

    def _gen_sample_data(self):
        # the following solves problems that arise during testing
        # as we have a mixed parametric, non-parametric distribution
        # here, supplying a feasible starting solution is non-trivial.
        # the data is part of the solution!
        sample_quantiles = np.linspace(0.001, .999, 500)
        sample_data = self.dist.ppf(sample_quantiles,
                                    *self.dist._feasible_start)
        rain_mask = sample_data > self.thresh
        sample_data[np.argsort(sample_data)[-5:-1]] *= 2
        sample_data[~rain_mask] = 0
        return np.sort(sample_data)

    def _pdf(self, x, q_thresh, rain_prob, kernel_width, kernel_data,
           f_thresh, q_kde_eval=None, x_eval=None, *args, **kwds):
        (x,
         q_thresh,
         rain_prob,
         kernel_width,
         f_thresh) = np.atleast_1d(x,
                                   q_thresh,
                                   rain_prob,
                                   kernel_width,
                                   f_thresh)
        assert np.all(np.isfinite(rain_prob))
        if len(rain_prob) == 1:
            rain_prob = np.broadcast_to(rain_prob, x.shape)
        if len(kernel_width) == 1:
            kernel_width = np.broadcast_to(kernel_width, x.shape)
        if (np.squeeze(kernel_data).ndim == 1
                and kernel_data.dtype != np.dtype("O")):
            kernel_data = np.broadcast_to(kernel_data,
                                          (x.size, kernel_data.size))
        if len(q_thresh) == 1:
            q_thresh = np.broadcast_to(q_thresh, x.shape)
        if len(f_thresh) == 1:
            f_thresh = np.broadcast_to(f_thresh, x.shape)
        if x.ndim > 1 and rain_prob.ndim > 1:
            (x,
             rain_prob,
             f_thresh,
            ) = np.broadcast_arrays(x,
                                    rain_prob,
                                    f_thresh)

        rain_mask = x > self.thresh
        par_mask = (x < f_thresh) & rain_mask
        kde_mask = x >= f_thresh
        # the following tries to catch broadcast ambiguities, but might not
        # be the most parsimonious formulation in terms of memory
        dens = np.zeros_like(rain_prob + rain_mask)
        dens[~rain_mask] = rain_prob[~rain_mask]

        def par_scale(x):
            return ((x - self.thresh) / (f_thresh[par_mask] - self.thresh))

        # parametric part
        if np.any(par_mask):
            x_par = x[par_mask]
            x_par = par_scale(x_par)
            kwds_par = self._mask_kwds(par_mask, kwds)
            dens_par = self.dist.pdf(x_par, *args, **kwds_par)
            if dens.shape == par_mask.shape:
                try:
                    dens[par_mask] = np.atleast_1d(dens_par)
                except TypeError:
                    dens[par_mask] = np.atleast_1d(dens_par)[0]
            else:
                dens[:, par_mask[0]] = dens_par

        def kde_single(dens, x_kde, mask, kernel_width, kernel_data):
            dens_kde = kde.kernel_density(kernel_width,
                                          np.log(kernel_data),
                                          eval_points=np.log(x_kde))
            # dividing by x_kde because of chain-rule differentiation
            # of ln(x)
            dens[mask] = (1. - self.q_thresh) * dens_kde / x_kde
            dens[mask] /= kernel_data.size - 1

        # non-parametric (KDE) part
        if np.any(kde_mask):
            if kde_mask.ndim == 2:
                for tt, mask in enumerate(kde_mask):
                    kde_single(dens[tt], x[tt, mask], mask,
                               kernel_width[tt],
                               kernel_data[tt, 0])
            else:
                kde_single(dens, x[kde_mask], kde_mask,
                           kernel_width[kde_mask],
                           kernel_data)
        return dens

    def _cdf(self, x, q_thresh, rain_prob, kernel_width, kernel_data,
           f_thresh, q_kde_eval=None, x_eval=None, *args, **kwds):
        (x,
         q_thresh,
         rain_prob,
         kernel_width,
         f_thresh) = np.atleast_1d(x,
                                   q_thresh,
                                   rain_prob,
                                   kernel_width,
                                   f_thresh)
        # i want to manipulate kwds, but only here
        kwds = {k: v for k, v in kwds.items()}
        if len(rain_prob) == 1:
            rain_prob = np.broadcast_to(rain_prob, x.shape)
        if len(kernel_width) == 1:
            kernel_width = np.broadcast_to(kernel_width, x.shape)
        if kernel_data.ndim == 1 and kernel_data.dtype != np.dtype("O"):
            kernel_data = np.broadcast_to(kernel_data,
                                          (len(x), len(kernel_data)))
        if len(q_thresh) == 1:
            q_thresh = np.broadcast_to(q_thresh, x.shape)
        if len(f_thresh) == 1:
            f_thresh = np.broadcast_to(f_thresh, x.shape)
        upper_eval = np.atleast_1d(kwds.pop("u", [None]))
        if len(upper_eval) == 1:
            upper_eval = np.broadcast_to(upper_eval, x.shape)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rain_mask = x > self.thresh
            kde_mask = x >= f_thresh
            par_mask = ~kde_mask & rain_mask
        # fill with middle quantile of non-rain
        qq = np.zeros_like(x, dtype=float)
        n_non_rain = (~rain_mask).sum()
        if n_non_rain > 0:
            qq[~rain_mask] = ((1 - rain_prob[~rain_mask])
                              * np.random.random(n_non_rain))

        if self.debug:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.hist(qq[~rain_mask], 40, histtype="step",
                    label="norain")
        # parametric part
        if np.any(par_mask):
            x_par = x[par_mask]
            kwds_par = self._mask_kwds(par_mask, kwds)
            x_par = ((x_par - self.thresh) /
                     (f_thresh[par_mask] - self.thresh))
            if "l" in kwds_par:
                x_par = np.maximum(x_par, kwds_par["l"])
            if self.debug:
                assert np.all(x_par >= 0)
                assert np.all(x_par <= 1)
            if "l" not in kwds_par:
                kwds_par["l"] = 0.
            if "u" not in kwds_par:
                kwds_par["u"] = 1.
            q_par = self.dist.cdf(x_par, *args, **kwds_par)
            p0 = (1 - rain_prob)[par_mask]
            p1 = q_thresh[par_mask]
            q_par = p0 + (p1 - p0) * q_par
            qq[par_mask] = q_par
            assert np.all(np.isfinite(q_par))

        # non-parametric (KDE) part
        if np.any(kde_mask):
            assert np.all(kernel_width[kde_mask] > 0)
            if q_kde_eval.ndim == 2:
                q_kde_eval = q_kde_eval[kde_mask]
            if x_eval.ndim == 2:
                x_eval = x_eval[kde_mask]
            q_kde = self._kde_cdf(x[kde_mask],
                                  q_kde_eval,
                                  x_eval)
            qq[kde_mask] = q_kde
            if self.debug:
                assert np.all(np.isfinite(q_kde))
            if self.debug:
                ax.hist(q_kde, 20, histtype="step", label="kde")
                ax.legend()
            assert np.all(np.isfinite(q_kde))
        return qq

    def _ppf(self, qq, q_thresh, rain_prob, kernel_width, kernel_data,
           f_thresh, q_kde_eval=None, x_eval=None, *args, **kwds):
        (qq,
         q_thresh,
         rain_prob,
         kernel_width,
         f_thresh) = np.atleast_1d(qq,
                                   q_thresh,
                                   rain_prob,
                                   kernel_width,
                                   f_thresh)
        assert np.all(np.isfinite(qq))
        # i want to manipulate kwds, but only here
        kwds = {k: v for k, v in kwds.items()}
        if len(rain_prob) == 1:
            rain_prob = np.broadcast_to(rain_prob, qq.shape)
        if len(kernel_width) == 1:
            kernel_width = np.broadcast_to(kernel_width, qq.shape)
        if kernel_data.ndim == 1 and kernel_data.dtype != np.dtype("O"):
            kernel_data = np.broadcast_to(kernel_data,
                                          (len(qq), len(kernel_data)))
        if len(q_thresh) == 1:
            q_thresh = np.broadcast_to(q_thresh, qq.shape)
        if len(f_thresh) == 1:
            f_thresh = np.broadcast_to(f_thresh, qq.shape)
        upper_eval = np.atleast_1d(kwds.pop("u", [None]))
        if len(upper_eval) == 1:
            upper_eval = np.broadcast_to(upper_eval, qq.shape)

        # we want to have zero rain where probability of rain is lower
        # than rain_prob
        x = np.zeros_like(qq, dtype=float)
        rain_mask = qq > 1 - rain_prob
        kde_mask = qq >= q_thresh
        par_mask = rain_mask & ~kde_mask
        if np.all(~rain_mask):
            return x

        # parametric part
        if np.any(par_mask):
            kwds_par = self._mask_kwds(par_mask, kwds)
            q_par = qq[par_mask]
            p0 = 1 - rain_prob[par_mask]
            p1 = q_thresh[par_mask]
            q_par = (q_par - p0) / (p1 - p0)
            x_par = self.dist.ppf(q_par, *args, **kwds_par)
            if self.debug:
                assert np.all(q_par >= 0)
                assert np.all(q_par <= 1)
            x_par = ((f_thresh[par_mask] - self.thresh) * x_par
                     + self.thresh)
            x[par_mask] = x_par
            assert np.all(np.isfinite(x_par))

        # non-parametric (KDE) part
        if np.any(kde_mask):
            if q_kde_eval.ndim == 2:
                q_kde_eval = q_kde_eval[kde_mask]
            if x_eval.ndim == 2:
                x_eval = x_eval[kde_mask]
            x_kde = self._kde_ppf(qq[kde_mask],
                                  q_kde_eval,
                                  x_eval)
            x[kde_mask] = x_kde
            assert np.all(np.isfinite(x_kde))
        return x

    def _fit_qthresh(self, q_thresh, x, x0=None, *args, **kwds):
        rain_mask = x > self.thresh
        rain_prob = np.mean(rain_mask)
        self.q_thresh = q_thresh
        f_thresh = max(np.percentile(x, 100 * self.q_thresh),
                       self.thresh)
        kde_mask = x >= f_thresh
        par_mask = rain_mask & ~kde_mask
        if np.sum(par_mask) > 1:
            rain_kwds = self._mask_kwds(par_mask, kwds)
            x_par = x[rain_mask & ~kde_mask]
            x_par = ((x_par - self.thresh) /
                     (f_thresh - self.thresh))
            par_names = self.dist.parameter_names
            if "u" in par_names:
                # the supplied upper limit is not meant for the
                # parametric but for the kde part
                rain_kwds["u"] = 1.
            par_solution = list(self.dist.fit(x_par, **rain_kwds))
        else:
            par_solution = self.dist._feasible_start
        self.success = True     # haha!
        kernel_data = x[kde_mask]
        if len(kernel_data) > 0:
            kernel_width = self.fit_kde(np.log(kernel_data))
        else:
            kernel_width = None
        upper_eval = kwds.pop("u", None)
        q_kde_eval, x_eval_log = self._kde_integral(kernel_width,
                                                    kernel_data,
                                                    f_thresh,
                                                    upper_eval)
        x_eval = np.exp(x_eval_log)
        self.fitted_pars = ((q_thresh, rain_prob, float(kernel_width),
                             kernel_data, f_thresh, q_kde_eval,
                             x_eval) + tuple(par_solution))
        self.q_thresh = q_thresh
        density = self.pdf(x, *self.fitted_pars)
        mask = (density > 0) & np.isfinite(density)
        lik = -np.nansum(np.log(density[mask]))
        return lik

    def _fit(self, x, x0=None, *args, **kwds):
        result = sp_optimize.minimize_scalar(
            self._fit_qthresh,
            args=(x, x0) + args,
            bounds=[self.q_thresh_lower,
                    self.q_thresh_upper],
            method="bounded")
        return self.fitted_pars

    def fit_ml(self, x, x0=None, *args, **kwds):
        """This is a lie, we don't fit rainmix with ML."""
        params = self._fit(x, x0, *args, **kwds)
        params = list(params)
        supplements = {name: params[self.parameter_names.index(name)]
                       for name in self.supplements_names}
        fixed_names = [name for name in kwds.keys()
                       if name in self.parameter_names]
        params = [param
                  for name, param in zip(self.parameter_names,
                                      params)
                  if
                  (name not in self.supplements_names) and
                  (name not in fixed_names)]
        return self.Result(x=params,
                           supplements=supplements,
                           success=self.success)

    def _constraints(self, x, q_thresh, rain_prob, kernel_width,
                   kernel_data, f_thresh, q_kde_eval=None,
                   x_eval=None, **kwds):
        (rain_prob,
         kernel_width,
         kernel_data,
         f_thresh) = map(np.asarray, (rain_prob, kernel_width,
                                      kernel_data, f_thresh))
        mask = ((q_thresh < self.q_thresh_lower) |
                (q_thresh > self.q_thresh_upper) |
                (rain_prob < 0) |
                (rain_prob > 1) |
                (kernel_width <= 0) |
                (f_thresh < 0))
        mask |= self.dist._constraints(x, **kwds)
        return mask

    def _invalid_x(self, x, *args, **kwds):
        mask = np.atleast_1d(np.full_like(x, False, dtype=bool))
        return mask

# q_threshold = .75
q_threshold = .9
# rainmix_kumaraswamy = RainMix(kumaraswamy, threshold=.001,
#                               q_threshold=q_threshold)
rainmix_kumaraswamy = RainMix(kumaraswamy, threshold=.001)
# rainmix_expon = RainMix(expon, q_threshold=q_threshold)
# rainmix_weibull = RainMix(weibull, q_threshold=q_threshold)
# rainmix_gamma = RainMix(gamma, q_threshold=q_threshold)
# rainmix_lognormal = RainMix(lognormal, q_threshold=q_threshold)
# rainmix_gamma1 = RainMix(gamma1)


class Rain(_Rain):
    def __init__(self, distribution, threshold=.0001):
        self.dist = distribution
        self.thresh = threshold
        self.parameter_names = tuple(["rain_prob"] +
                                     list(distribution.parameter_names))
        self.n_pars = len(self.parameter_names)
        self._feasible_start = (0.98, ) + self.dist._feasible_start
        self.name = "rain " + distribution.name

    def _pdf(self, x, rain_prob, *args, **kwds):
        x, rain_prob = np.atleast_1d(x, rain_prob)
        rain_mask = x >= self.thresh
        if len(rain_prob) == 1:
            rain_prob = np.broadcast_to(rain_prob, x.shape)
        # the following tries to catch broadcast ambiguities, but might not
        # be the most parsimonious formulation in terms of memory
        dens = np.empty_like(rain_prob + rain_mask, dtype=float)
        dens[~rain_mask] = rain_prob[~rain_mask]
        kwds_rain = self._mask_kwds(rain_mask, kwds)
        dens_rain = (self.dist.pdf(x[rain_mask] - self.thresh, *args,
                                   **kwds_rain)
                     * rain_prob[rain_mask])
        if dens.shape == rain_mask.shape:
            dens[rain_mask] = dens_rain
        else:
            dens[:, rain_mask[0]] = dens_rain
        return dens

    def _cdf(self, x, rain_prob, *args, **kwds):
        x, rain_prob = np.atleast_1d(x, rain_prob)
        qq = np.zeros_like(x, dtype=float)
        if len(rain_prob) == 1:
            rain_prob = np.full_like(x, rain_prob)
        finite_mask = np.isfinite(x)
        x_finite = np.where(finite_mask, x, 0)
        rain_mask = x_finite >= self.thresh
        # subtracting self.thresh from x has the unwanted side-effect of
        # shifting x to possibly out-of-bound values. so this should not be
        # done with values where ~rain_mask.
        # as we mask x, we also have to mask the parameters given in **kwds
        kwds_rain = self._mask_kwds(rain_mask, kwds)
        p0 = 1 - rain_prob[rain_mask]
        qq[rain_mask] = (p0
                         + rain_prob[rain_mask]
                         * self.dist.cdf(x[rain_mask] - self.thresh,
                                         *args, **kwds_rain))
        qq[~rain_mask] = 1 - rain_prob[~rain_mask]
        qq[~finite_mask] = np.nan
        return qq

    def _ppf(self, qq, rain_prob, *args, **kwds):
        qq, rain_prob = np.atleast_1d(qq, rain_prob)
        if len(rain_prob) == 1:
            rain_prob = np.full_like(qq, rain_prob)
        # we want to have zero rain, where probability of rain is lower than
        # rain_prob
        x = np.zeros_like(qq, dtype=float)
        rain_mask = qq > 1 - rain_prob
        if np.all(~rain_mask):
            return x
        # subtracting self.thresh from x has the unwanted side-effect of
        # shifting x to possibly out-of-bound values. so this should not be
        # done with values where ~rain_mask.
        # as we mask x, we also have to mask the parameters given in **kwds
        kwds_rain = self._mask_kwds(rain_mask, kwds)
        x[rain_mask] = (self.thresh +
                        self.dist.ppf((qq[rain_mask] - 1.) /
                                      rain_prob[rain_mask] + 1,
                                      *args, **kwds_rain))
        return x

    def _fit(self, x, **kwds):
        rain_mask = x >= self.thresh
        rain_prob = np.mean(rain_mask)
        solution = self.dist.fit(x[rain_mask] - self.thresh, **kwds)
        return tuple([rain_prob] + list(solution))


rain_expon = Rain(expon, threshold=.002)
rain_weibull = Rain(weibull, threshold=.002)

if __name__ == "__main__":
    import vg
    from vg import vg_base, vg_plotting
    # import config_konstanz_disag as conf
    import config_konstanz as conf
    vg.set_conf(conf)
    # times_hourly, met = vg.read_met(vg.conf.met_file)
    # rain, times = vg.my.sumup(met["R"], 24, times_hourly)
    met_vg = vg.VG(("R", "theta", "ILWR", "Qsw", "rh"),
                   verbose=True, refit="R")
    rain_dist, solution = met_vg.dist_sol["R"]
    met_vg.simulate()
    # rain = met_vg.data_raw[0]
    # rain_dist = rainmix_expon
    # rain_dist = vg.sd.SlidingDist(rain_dist, rain, met_vg.times, verbose=True)

    # solution = rain_dist.fit()
    # rain_dist.plot_seasonality_fit()
    # rain_dist.plot_monthly_params()
    rain_dist.plot_fourier_fit()
    rain_dist.plot_monthly_fit(solution,
                               # dists_alt=(Rain(gamma, threshold=.015),
                               #            expon)
                               )
    met_vg.plot_monthly_hists("R")
    met_vg.plot_meteogramm_daily()
    # rain_dist.plot_monthly_params()
    quantiles = rain_dist.cdf(solution)
    my.hist(quantiles, 20)
    my.hist(met_vg.data_trans[0], 20, dist=norm)
    # rain_dist.scatter_pdf(solution)
    plt.show()
