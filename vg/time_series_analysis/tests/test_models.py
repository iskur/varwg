from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import os
import numpy as np
from numpy.testing import (assert_almost_equal, TestCase, run_module_suite,
                           decorators)
from vg.time_series_analysis import models

# the following parameters and data are taken from p.707 in order to test
# functions regarding the VAR least-squares estimator
B_test = np.matrix([
    [-.017, -.320, .146, .961, -.161, .115, .934],
    [.016, .044, -.153, .289, .050, .019, -.010],
    [.013, -.002, .225, -.264, .034, .355, -.022]])
Bex_test = np.matrix([
    [-.017, -.320, .146, .961, -.161, .115, .934, .25],
    [.016, .044, -.153, .289, .050, .019, -.010, .1],
    [.013, -.002, .225, -.264, .034, .355, -.022, -.25]])
A_test = np.matrix([
    [-.319, .147, .959, -.160, .115, .932],
    [.044, -.152, .286, .050, .020, -.012],
    [-.002, .225, -.264, .034, .355, -.022]])
sigma_u_test = 1e-4 * np.matrix([[21.3, .72, 1.23],
                                 [.72, 1.37, .61],
                                 [1.23, .61, .89]])
VAR_p = 2
VAR_K = 3
VARMA_p, VARMA_q = 2, 1
e1_filepath = os.path.join(os.path.dirname(__file__), "e1.dat")
data = np.loadtxt(e1_filepath, skiprows=7).T
data = np.log(data[:, 1:]) - np.log(data[:, :-1])
data = data[:, :75]
data_means = data.mean(axis=1).reshape((3, 1))


class Test(TestCase):
    msg = "They do suggest the right p, however."
    # Test is not conclusive: solutions are for ML estimator of VAR process...
    # We get however, the same p!

    @decorators.setastest(False)
    @decorators.knownfailureif(True, msg)
    def test_AIC(self):
        """See p.148"""
        order_data = np.copy(data[:, 4:])
        p = 1
        K, T = order_data.shape
        sigma_u = (float(T - K * p - 1) / T) * models.VAR_LS(order_data, p)[1]
        print(np.linalg.det(sigma_u) * 1e11)
        print(models.VAR_LS(order_data, p)[1] * 1e4)
        objectives = np.array([(old_div(float(T - K * p_ - 1), T)) *
                               models.AIC(models.VAR_LS(order_data, p_)[1], p_,
                                          order_data.shape[1])
                               for p_ in range(5)])
        AICs = -24 - np.array([.42, .5, .59, .41, .36])
        assert_almost_equal(AICs, objectives, decimal=2)

    @decorators.setastest(False)
    @decorators.knownfailureif(True, msg)
    def test_HQ(self):
        order_data = np.copy(data[:, 4:])
        objectives = np.array([models.HQ(models.VAR_LS(order_data, p)[1], p,
                                         order_data.shape[1])
                               for p in range(5)])
        HQs = -24 - np.array([.42, .38, .37, .07, -.1])
        assert_almost_equal(HQs, objectives, decimal=2)

    @decorators.setastest(False)
    @decorators.knownfailureif(True, msg)
    def test_SC(self):
        order_data = np.copy(data[:, 4:])
        objectives = np.array([models.SC(models.VAR_LS(order_data, p)[1], p,
                                         order_data.shape[1])
                               for p in range(5)])
        SCs = np.array([-24.42, -24.21, -24.02, -23.55, -23.21])
        assert_almost_equal(SCs, objectives, decimal=2)

    @decorators.setastest(False)
    @decorators.knownfailureif(True, msg)
    def test_FPE(self):
        order_data = np.copy(data[:, 4:])
        K, T = order_data.shape
        objectives = np.array([(old_div(float(T - K * p - 1), T)) *
                               models.FPE(models.VAR_LS(order_data, p)[1], p,
                                          T)
                               for p in range(5)])
        objectives *= 1e11
        FPEs = np.array([2.691, 2.5, 2.272, 2.748, 2.91])
        assert_almost_equal(FPEs, objectives, decimal=2)

    def test__scale_additive(self):
        self.assertRaises(ValueError, models._scale_additive,
                          [1, 0],
                          [[.5, 0],
                           [0, 1],
                           [0, 0]])

        scaled = models._scale_additive([1, 0],
                                        [[.5, 0],
                                         [0, 1]])
        self.assertAlmostEquals(tuple(scaled), (.5, 0))

        scaled = models._scale_additive([1, 0],
                                        [[.5, 0, .5, 0],
                                         [0, 1, 0, 1]], p=1)
        self.assertAlmostEquals(tuple(scaled), (.5, 0))

        scaled = models._scale_additive([[1, .5], [0, 0]],
                                        [[.5, 0],
                                         [0, 1]])
        assert_almost_equal(scaled, np.asmatrix([[.5, .25],
                                                 [0, 0]]))
        self.assertEquals(scaled.shape, (2, 2))

    def test_VAR_LS(self):
        """Checking the example mentioned in the VAR_LS-docstring (p. 707f)."""
        # values from 1960-1978 with presample data
        B, sigma_u = models.VAR_LS(data, VAR_p)
        # sigma_u = np.cov(models.VAR_LS_residuals(data, B, VAR_p), ddof=1)
        assert_almost_equal(B, B_test, decimal=3)
        assert_almost_equal(sigma_u, sigma_u_test, decimal=6)

    # def test_VAR_YW(self):
    #     """Checking the example mentioned in the VAR_LS-docstring (p. 707f)."""
    #     # values from 1960-1978 with presample data
    #     A, sigma_u = models.VAR_YW(data, VAR_p)
    #     print "\n", A
    #     sigma_u = np.cov(models.VAR_LS_residuals(data, B, VAR_p), ddof=1)
    #     assert_almost_equal(A_test, A, decimal=2)

    def test_VAR_LS_sim(self):
        """Does VAR_LS_sim reproduce the correlation matrix?"""
        T = 500
        sim = models.VAR_LS_sim(B_test, sigma_u_test, T)
        assert_almost_equal(np.corrcoef(data), np.corrcoef(sim),
                            decimal=1)

    def test_VAR_LS_sim_fixed_var(self):
        """Does VAR_LS_sim return the fixed values given via fixed_data?"""
        T = 5
        fixed = np.nan * np.empty(VAR_K * T).reshape((VAR_K, T))
        fixed[0] = np.array([0, 0, 1., 0, 0])
        sim = models.VAR_LS_sim(B_test, sigma_u_test, T, fixed_data=fixed)
        assert_almost_equal(sim[0], fixed[0])

    def test_VARMA_LS_sim_fixed_var(self):
        """Does VAR_LS_sim return the fixed values given via fixed_data?"""
        T = 5
        fixed = np.nan * np.empty(VAR_K * T).reshape((VAR_K, T))
        fixed[0] = np.array([0, 0, 1., 0, 0])
        AM, sigma_u = models.VARMA_LS_prelim(data, VARMA_p, VARMA_q)[:-1]
        sim = models.VARMA_LS_sim(AM, VARMA_p, VARMA_q, sigma_u, data_means, T,
                                  fixed_data=fixed)
        assert_almost_equal(sim[0], fixed[0])

    #  def test_VARMA_LS_sim(self):
    #      """Does VAR_LS_sim reproduce the correlation matrix?"""
    #      T = data.shape[1] * 100
    #      AM, sigma_u = models.VARMA_LS_prelim(data, VARMA_p, VARMA_q)[:-1]
    #      means = data.mean(axis=1)
    #      sim = models.VARMA_LS_sim(AM, VARMA_p, VARMA_q, sigma_u, means, T)
    #      assert_almost_equal(np.corrcoef(data), np.corrcoef(sim), decimal=2)

    # def test_VARMA_LS_sim_VARMA_residuals(self):
    #     """Round-trip test: do we get predifined residuals back from
    #     VARMA_LS_residuals when simulating with VARMA_LS_sim?"""
    #     T = 5
    #     residuals_test = np.asmatrix(
    #                         [np.random.multivariate_normal(VAR_K * [0],
    #                                                        sigma_u_test)
    #                          for t in xrange(T)]).reshape((VAR_K, T))
    #     AM, sigma_u = models.VARMA_LS_prelim(data, VARMA_p, VARMA_q)[:-1]
    #     sim = models.VARMA_LS_sim(AM, VARMA_p, VARMA_q, sigma_u, T,
    #                               residuals_test, n_sim_multiple=1)
    #     residuals = models.VARMA_residuals(sim, AM, VARMA_p, VARMA_q)
    #     assert_almost_equal(residuals, residuals_test, decimal=3)

    # def test_MGARCH_sim_MGARCH_residuals(self):
    #     np.random.seed(0)
    #     # B = [[-.24e-3, -.00, 0.02, -.18, .16, -.08, .11],
    #     #      [-.42e-3, 0.12, -.13, -.08, .03, -.01, .01]]
    #     # sigma_u = [[1., .8],
    #     #            [.8, 1.]]
    #     T = 1000
    #     sigma_z = .5
    #     a0, a1 = .25, .5
    #     sigma_t = np.ones((2, T))
    #     ut = np.zeros((2, T))
    #     zts = sigma_z * np.random.multivariate_normal([0, 0],
    #                                                   [[1., .8],
    #                                                    [.8, 1.]],
    #                                                   T)
    #     zts = zts.T
    #     for t in xrange(1, T):
    #         sigma_t[:, t] = np.sqrt(a0 + a1 * ut[:, t - 1] ** 2)
    #         ut[:, t] = sigma_t[:, t] * zts[:, t]
    #     ut = np.asmatrix(ut)

    #     from vg.time_series_analysis import time_series as ts

    #     # ts.plt.plot(zts.T)
    #     # ts.plt.plot(ut.T)
    #     # ts.plot_auto_corr(ut, k_range=7)
    #     # ts.plt.show()

    #     params = models.MGARCH_ML(ut, 2, 2)
    #     ut_sim = models.MGARCH_sim(params, T, np.cov(ut))

    #     ts.plt.plot_auto_corr([ut.T, ut_sim.T], k_range=7)
    #     ts.plt.show()

    def test_VAR_LS_sim_VAR_residuals(self):
        """Round-trip test: do we get predifined residuals back from
        VAR_LS_residuals when simulating with VAR_LS_sim?"""
        T = 5
        residuals_test = np.asmatrix(
                            [np.random.multivariate_normal(VAR_K * [0],
                                                           sigma_u_test)
                             for t in range(T)]).reshape((VAR_K, T))
        sim = models.VAR_LS_sim(B_test, sigma_u_test, T, u=residuals_test,
                                n_presim_steps=0)
        residuals = models.VAR_residuals(sim, B_test, VAR_p)
        assert_almost_equal(residuals, residuals_test)

    def test_VAREX_LS_sim_VAREX_residuals(self):
        """Round-trip test: do we get predifined residuals back from
        VAREX_LS_residuals when simulating with VAREX_LS_sim?"""
        T = 5
        residuals_test = np.asmatrix(
                            [np.random.multivariate_normal(VAR_K * [0],
                                                           sigma_u_test)
                             for t in range(T)]).reshape((VAR_K, T))
        ex = np.full(T, .1)
        sim, ex_out = models.VAREX_LS_sim(Bex_test, sigma_u_test, T, ex,
                                          u=residuals_test, n_presim_steps=0)
        residuals = models.VAREX_residuals(sim, ex, Bex_test, VAR_p)
        assert_almost_equal(residuals, residuals_test)
        assert_almost_equal(ex_out, ex)

    def test_VAREX_LS_sim_VAREX_residuals_funcy(self):
        """Round-trip test: do we get predifined residuals back from
        VAREX_LS_residuals when simulating with VAREX_LS_sim?"""
        T = 5
        residuals_test = np.asmatrix(
                            [np.random.multivariate_normal(VAR_K * [0],
                                                           sigma_u_test)
                             for t in range(T)]).reshape((VAR_K, T))
        ex_kwds = dict(fac=1.5)
        ex = lambda x, fac: np.mean(x[:, -1]) * fac
        sim, ex_out = models.VAREX_LS_sim(Bex_test, sigma_u_test, T, ex,
                                          u=residuals_test, n_presim_steps=0,
                                          ex_kwds=ex_kwds)
        residuals = models.VAREX_residuals(sim, ex, Bex_test, VAR_p, ex_kwds)
        assert_almost_equal(residuals, residuals_test)

    def test_SVAR_LS_sim_SVAR_residuals(self):
        """Round-trip test: do we get predifined residuals back from
        SVAR_LS_residuals when simulating with SVAR_LS_sim?"""
        T = 365
        residuals_test = np.asmatrix(
                            [np.random.multivariate_normal(VAR_K * [0],
                                                           sigma_u_test)
                             for t in range(T)]).reshape((VAR_K, T))
        doys = np.arange(1, T + 1)
        # have to have seasonal B_test and sigma_u_test
        Bs_test = np.empty((B_test.shape[0], B_test.shape[1], T))
        sigma_u_test_s = np.empty((sigma_u_test.shape[0],
                                   sigma_u_test.shape[1], T))
        Bs_test[...] = ((np.arange(T) % 2 - .5)[None, None, :] * 2 *
                        np.asarray(B_test[..., None]))
        sigma_u_test_s[:] = sigma_u_test[..., None]
#        residuals_test[:] = 0
        sim = models.SVAR_LS_sim(Bs_test, sigma_u_test_s, doys,
                                 u=residuals_test, n_presim_steps=0)
        residuals = models.SVAR_residuals(sim, doys, Bs_test, VAR_p)
        # the beginning time steps will not be matched, but I do not care!
        assert_almost_equal(residuals[:, VAR_p:], residuals_test[:, VAR_p:],
                            decimal=6)

if __name__ == "__main__":
    run_module_suite()
