from __future__ import division, print_function

import sys

import numpy as np
from scipy import integrate
from scipy.misc import derivative
import numpy.testing as npt
import matplotlib.pyplot as plt

import vg.time_series_analysis.distributions as ds
from vg import helpers as my


distributions = {name: obj for name, obj in list(ds.__dict__.items())
                 if isinstance(obj, ds.Dist)}
if "SkewNormal" in distributions and not ds.owens:
    del distributions["SkewNormal"]

# import pandas as pd
# from vg import config_template
# # the rainmix distribution requires data to be initialized
# sample_met_filepath = config_template.met_file
# rain_pd = pd.read_table(sample_met_filepath, sep="\t", usecols="R")
# rain_values = np.squeeze(rain_pd[:1000].as_matrix())
# rainmix_expon = ds.RainMix(ds.expon, rain_values)
# distributions["rainmix expon"] = rainmix_expon

dists_frozen = {name: dist(*dist._feasible_start)
                for name, dist in list(distributions.items())}
# avoid quantiles of 0 or 1, because that causes problems with the unbounded
# distributions
quantiles = np.linspace(1e-6, 1 - 1e-6, 11)
quantiles_more = np.linspace(0.001, .999, 500)
incr = 1e-6


class Test(npt.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.verbose = False

    @npt.dec.slow
    def test_fit(self):
        """Is fit able to reproduce parameters of a ppf-generated sample?"""
        if self.verbose:
            print()
        for dist_name, dist in list(distributions.items()):
            if self.verbose:
                print("\t", dist_name)
            known_params = dist._feasible_start
            x = dist.ppf(quantiles_more, *known_params)
            if hasattr(dist, "_additional_kwds"):
                kwds = dist._additional_kwds
            else:
                kwds = {}
            fitted_params = dist.fit(x, **kwds)
            if len(kwds) > 0:
                known_params = known_params[:-len(kwds)]
            if isinstance(dist, ds.RainMix):
                try:
                    for name, known, fitted in zip(dist.parameter_names,
                                                   known_params,
                                                   fitted_params):
                        if name in dist.supplements_names:
                            continue
                        npt.assert_almost_equal(known, fitted, decimal=1)
                except (ValueError, AssertionError):
                    if not self.verbose:
                        raise
                    import matplotlib.pyplot as plt

                    def rel_ranks(x):
                        n = len(x)
                        return (np.arange(1, n + 1) - .5) / n

                    plt.plot(x, rel_ranks(x), "-o", label="x")
                    plt.plot(dist.sample_data, rel_ranks(dist.sample_data),
                             "-x", label="sample")
                    plt.legend(loc="best")
                    plt.title(dist.name)
                    plt.show()
                    print(2 * "\t", name)
                    raise
            else:
                try:
                    npt.assert_almost_equal(known_params,
                                            fitted_params,
                                            decimal=1)
                except AssertionError:
                    print("\rfailed")

    def test_pdf_integral(self):
        """Is the cdf the integral of the pdf?"""
        if self.verbose:
            print()
        for dist_name, dist_frozen in dists_frozen.items():
            if isinstance(dist_frozen.dist, ds.RainMix):
                # see specific test below
                continue
            if self.verbose:
                print("\t", dist_name)
            # get two feasible input values
            q0, q1 = .25, .99
            x0, x1 = dist_frozen.ppf([q0, q1])
            xx = np.linspace(x0, x1, 100)
            densities = dist_frozen.pdf(xx)
            integral = integrate.trapz(densities, xx)
            try:
                npt.assert_almost_equal(integral, q1 - q0, decimal=2)
            except AssertionError as exc:
                if not self.verbose:
                    raise
                dist_frozen.plot_fit(dist_frozen.sample(500))
                plt.show()
                print(exc)
                raise

    def test_rainmix_pdf(self):
        if self.verbose:
            print()
        rainmix_dists = {name: dist
                         for name, dist in dists_frozen.items()
                         if isinstance(dist.dist, ds.RainMix)}
        for dist_name, dist_frozen in rainmix_dists.items():
            if self.verbose:
                print("\t", dist_name)
            dist_frozen.dist.debug = True
            # get two feasible input values
            # q0, q1 = .01, .99
            q0, q1 = 1e-6, 1 - 1e-6
            xx = dist_frozen.ppf(np.linspace(q0, q1, 500))
            x0, x1 = xx[[0, -1]]
            f_thresh = dist_frozen.parameter_dict["f_thresh"]
            q_thresh = dist_frozen.dist.q_thresh
            npt.assert_almost_equal(dist_frozen.ppf(q_thresh), f_thresh,
                                    decimal=3)
            npt.assert_almost_equal(dist_frozen.cdf(f_thresh), q_thresh,
                                    decimal=4)
            integral_par = integrate.quad(dist_frozen.pdf, x0, f_thresh)[0]
            integral_kde = integrate.quad(dist_frozen.pdf, f_thresh, x1)[0]
            try:
                npt.assert_almost_equal(integral_par,
                                        q_thresh - q0,
                                        decimal=3)
            except AssertionError as exc:
                print("parametric part")
                print(exc)
                raise
            try:
                npt.assert_almost_equal(integral_kde,
                                        q1 - q_thresh,
                                        decimal=2)
            except AssertionError as exc:
                print("KDE part")
                print(exc)
                print(f"exp/act: {integral_kde / (q1 - q_thresh)} "
                      f"act/exp: {(q1 - q_thresh) / integral_kde}")
                raise
            integral = integrate.quad(dist_frozen.pdf, x0, x1)[0]
            try:
                npt.assert_almost_equal(integral, q1 - q0, decimal=2)
            except AssertionError as exc:
                # if not self.verbose:
                #     raise
                print("Full distribution")
                print(exc)
                sample = dist_frozen.sample(5000)
                fig, (ax1, ax2) = dist_frozen.plot_fit(sample)
                ax2.axhline(dist_frozen.dist.q_thresh,
                            linestyle="--")
                ax2.axvline(dist_frozen.parameter_dict["f_thresh"],
                            linestyle="--")
                # pdf by cdf differentiation
                cdf_diff = np.array([derivative(dist_frozen.cdf, x0,
                                                dx=1e-6)
                                     for x0 in xx])
                ax2.plot(xx, cdf_diff, "y--")
                ax2.set_ylim([0, 1])
                plt.show()
                raise

    def test_roundtrip(self):
        """Is the ppf really the inverse of the cdf?"""
        if self.verbose:
            print()
        for dist_name, dist_frozen in dists_frozen.items():
            dist_frozen.dist.debug = True
            if self.verbose:
                print("\t", dist_name)
            x = dist_frozen.ppf(quantiles_more)
            quantiles_back = dist_frozen.cdf(x)
            try:
                # distributions with lower thresholds will fail to
                # reproduce quantiles below it
                rain_prob = dist_frozen.parameter_dict["rain_prob"]
                dry_mask = quantiles_more < (1 - rain_prob)
                quantiles_back[dry_mask] = quantiles_more[dry_mask]
            except KeyError:
                pass
            npt.assert_array_less(quantiles_back, 1.)
            npt.assert_array_less(0, quantiles_back + sys.float_info.min)
            try:
                # if isinstance(dist_frozen.dist, ds.RainMix):
                #     quantiles_back -= 1e-4
                npt.assert_almost_equal(quantiles_more,
                                        quantiles_back, decimal=3)
            except AssertionError:
                if not self.verbose:
                    raise
                import matplotlib.pyplot as plt

                def rel_ranks(x):
                    n = len(x)
                    return (np.arange(n) - .5) / n

                fig, axs = plt.subplots(ncols=2,
                                        sharey=True,
                                        figsize=(8, 4),
                                        # subplot_kw=dict(aspect="equal")
                                        )
                axs[0].scatter(quantiles_more, quantiles_back, marker="x")
                if isinstance(dist_frozen.dist, ds.RainMix):
                    dist = dist_frozen.dist
                    axs[1].plot(x, rel_ranks(x), "-x", label="x")
                    axs[1].plot(dist.sample_data, rel_ranks(dist.sample_data),
                                "-x", label="sample_data")
                    axs[1].legend(loc="best")
                    axs[0].axvline(dist.q_thresh)
                    axs[0].axhline(dist.q_thresh)
                    axs[1].axhline(dist.q_thresh)
                    axs[1].axvline(dist_frozen.parameter_dict["f_thresh"])
                    # sample_quantiles = dist_frozen.cdf(dist.sample_data)
                    # ax.scatter(quantiles_more, sample_quantiles,
                    #            marker="o",
                    #            # facecolor=4 * (0,),
                    #            # edgecolor=4 * (1,),
                    # )
                axs[0].plot([0, 1], [0, 1])
                axs[0].set_xlabel("quantiles_more")
                axs[0].set_ylabel("quantiles_back")
                axs[0].grid(True)
                axs[1].grid(True)
                fig.suptitle(dist_frozen.name)
                plt.show()
                raise

    def test_sample(self):
        if self.verbose:
            print()
        for dist_name, dist_frozen in list(dists_frozen.items()):
            if self.verbose:
                print("\t", dist_name)
            sample_size = 100
            sample = dist_frozen.sample(sample_size)
            npt.assert_equal(sample.size, sample_size)

    def test_scalar(self):
        """Do pdf, cdf and ppf accept and return scalar values?"""
        if self.verbose:
            print()
        for dist_name, dist_frozen in list(dists_frozen.items()):
            if isinstance(dist_name, ds._KDE):
                continue
            if self.verbose:
                print("\t", dist_name)
            self.assert_(np.isscalar(dist_frozen.pdf(1)))
            self.assert_(np.isscalar(dist_frozen.cdf(1)))
            self.assert_(np.isscalar(dist_frozen.ppf(.5)))

    def test_out_of_bounds_pdf(self):
        """Do pdfs return nan for input outside the range?"""
        if self.verbose:
            print()
        for dist_name, dist_frozen in list(dists_frozen.items()):
            if isinstance(dist_frozen.dist, ds.RainMix):
                continue
            pars = dist_frozen.parameter_dict
            lower = pars.get("lc") or pars.get("l")
            upper = pars.get("uc") or pars.get("u")
            if lower:
                if self.verbose:
                    print("\t lower bound of ", dist_name)
                ret = dist_frozen.pdf(lower - incr)
                self.assert_(np.isnan(ret), ret)
            if upper:
                if self.verbose:
                    print("\t upper bound of ", dist_name)
                ret = dist_frozen.pdf(upper + incr)
                self.assert_(np.isnan(ret), ret)

    # def test_out_of_bounds_cdf(self):
    #     """Do cdfs return nan for input outside the range?"""
    #     if self.verbose:
    #         print()
    #     for dist_name, dist_frozen in list(dists_frozen.items()):
    #         if isinstance(dist_frozen.dist, ds.RainMix):
    #             continue
    #         pars = dist_frozen.parameter_dict
    #         lower = pars.get("lc") or pars.get("l")
    #         upper = pars.get("uc") or pars.get("u")
    #         if lower:
    #             if self.verbose:
    #                 print("\t lower bound of ", dist_name)
    #             ret = dist_frozen.cdf(lower - incr)
    #             self.assert_(np.isnan(ret), ret)
    #         if upper:
    #             if self.verbose:
    #                 print("\t upper bound of ", dist_name)
    #             ret = dist_frozen.cdf(upper + incr)
    #             self.assert_(np.isnan(ret), ret)

    # def test_out_of_bounds_ppf(self):
    #     """Do ppfs return nan for input outside the range?"""
    #     if self.verbose:
    #         print()
    #     for dist_name, dist_frozen in list(dists_frozen.items()):
    #         if isinstance(dist_frozen.dist, ds.RainMix):
    #             continue
    #         pars = dist_frozen.parameter_dict
    #         lower = pars.get("lc") or pars.get("l")
    #         upper = pars.get("uc") or pars.get("u")
    #         if lower:
    #             if self.verbose:
    #                 print("\t lower bound of ", dist_name)
    #             ret = dist_frozen.ppf(lower - incr)
    #             self.assert_(np.isnan(ret), ret)
    #         if upper:
    #             if self.verbose:
    #                 print("\t upper bound of ", dist_name)
    #             ret = dist_frozen.ppf(upper + incr)
    #             self.assert_(np.isnan(ret), ret)

    def test_rain(self):
        thresh = .1
        exp = ds.expon(0., 1.)
        exp_rv = exp.ppf(quantiles_more)
        # rain and no rain in it
        exp_rv = np.concatenate((np.zeros(len(quantiles_more) // 2), exp_rv))
        rain_dist = ds.Rain(ds.expon, thresh)
        rain = rain_dist(*rain_dist.fit(exp_rv))
        ranks = my.rel_ranks(exp_rv)
        rain_qq = rain.cdf(exp_rv)
        npt.assert_almost_equal(ranks[exp_rv > thresh],
                                rain_qq[exp_rv > thresh],
                                decimal=3)
        rain_retrans = rain.ppf(rain_qq)
        npt.assert_almost_equal(exp_rv[rain_retrans > thresh],
                                rain_retrans[rain_retrans > thresh])


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", RuntimeWarning)
    npt.run_module_suite()
