import os
import numpy as np
import numpy.testing as npt
import numpy.testing.decorators as decorators
import vg
from vg.time_series_analysis import models, tests

# the following parameters and data are taken from p.707 in order to test
# functions regarding the VAR least-squares estimator
B_test = np.array(
    [
        [-0.017, -0.320, 0.146, 0.961, -0.161, 0.115, 0.934],
        [0.016, 0.044, -0.153, 0.289, 0.050, 0.019, -0.010],
        [0.013, -0.002, 0.225, -0.264, 0.034, 0.355, -0.022],
    ]
)
Bex_test = np.array(
    [
        [-0.017, -0.320, 0.146, 0.961, -0.161, 0.115, 0.934, 0.25],
        [0.016, 0.044, -0.153, 0.289, 0.050, 0.019, -0.010, 0.1],
        [0.013, -0.002, 0.225, -0.264, 0.034, 0.355, -0.022, -0.25],
    ]
)
A_test = np.array(
    [
        [-0.319, 0.147, 0.959, -0.160, 0.115, 0.932],
        [0.044, -0.152, 0.286, 0.050, 0.020, -0.012],
        [-0.002, 0.225, -0.264, 0.034, 0.355, -0.022],
    ]
)
sigma_u_test = 1e-4 * np.array(
    [[21.3, 0.72, 1.23], [0.72, 1.37, 0.61], [1.23, 0.61, 0.89]]
)
VAR_p = 2
VAR_K = 3
VARMA_p, VARMA_q = 2, 1
e1_filepath = os.path.join(os.path.dirname(tests.__file__), "e1.dat")
data = np.loadtxt(e1_filepath, skiprows=7).T
data = np.log(data[:, 1:]) - np.log(data[:, :-1])
data = data[:, :75]
data_means = data.mean(axis=1).reshape((3, 1))


class Test(npt.TestCase):
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
        objectives = np.array(
            [
                (float(T - K * p_ - 1) / T)
                * models.AIC(
                    models.VAR_LS(order_data, p_)[1], p_, order_data.shape[1]
                )
                for p_ in range(5)
            ]
        )
        AICs = -24 - np.array([0.42, 0.5, 0.59, 0.41, 0.36])
        npt.assert_almost_equal(AICs, objectives, decimal=2)

    @decorators.setastest(False)
    @decorators.knownfailureif(True, msg)
    def test_HQ(self):
        order_data = np.copy(data[:, 4:])
        objectives = np.array(
            [
                models.HQ(
                    models.VAR_LS(order_data, p)[1], p, order_data.shape[1]
                )
                for p in range(5)
            ]
        )
        HQs = -24 - np.array([0.42, 0.38, 0.37, 0.07, -0.1])
        npt.assert_almost_equal(HQs, objectives, decimal=2)

    @decorators.setastest(False)
    @decorators.knownfailureif(True, msg)
    def test_SC(self):
        order_data = np.copy(data[:, 4:])
        objectives = np.array(
            [
                models.SC(
                    models.VAR_LS(order_data, p)[1], p, order_data.shape[1]
                )
                for p in range(5)
            ]
        )
        SCs = np.array([-24.42, -24.21, -24.02, -23.55, -23.21])
        npt.assert_almost_equal(SCs, objectives, decimal=2)

    @decorators.setastest(False)
    @decorators.knownfailureif(True, msg)
    def test_FPE(self):
        order_data = np.copy(data[:, 4:])
        K, T = order_data.shape
        objectives = np.array(
            [
                (float(T - K * p - 1) / T)
                * models.FPE(models.VAR_LS(order_data, p)[1], p, T)
                for p in range(5)
            ]
        )
        objectives *= 1e11
        FPEs = np.array([2.691, 2.5, 2.272, 2.748, 2.91])
        npt.assert_almost_equal(FPEs, objectives, decimal=2)

    def test_B2A(self):
        B = np.array([[0, 1, 2, 5, 6], [0, 3, 4, 7, 8]])
        A_exp = np.array(
            [[1, 2, 5, 6], [3, 4, 7, 8], [1, 0, 0, 0], [0, 1, 0, 0]]
        )
        A_act = models.B2A(B)
        npt.assert_almost_equal(A_act, A_exp)

    def test__scale_additive(self):
        self.assertRaises(
            ValueError,
            models._scale_additive,
            [1, 0],
            [[0.5, 0], [0, 1], [0, 0]],
        )

        scaled = models._scale_additive([1, 0], [[0.5, 0], [0, 1]])
        self.assertAlmostEqual(tuple(scaled), (0.5, 0))

        scaled = models._scale_additive(
            [1, 0], [[0.5, 0, 0.5, 0], [0, 1, 0, 1]], p=1
        )
        self.assertAlmostEqual(tuple(scaled), (0.5, 0))

        scaled = models._scale_additive([[1, 0.5], [0, 0]], [[0.5, 0], [0, 1]])
        npt.assert_almost_equal(scaled, np.array([[0.5, 0.25], [0, 0]]))
        self.assertEqual(scaled.shape, (2, 2))

    def test_VAR_LS(self):
        """Checking the example mentioned in the VAR_LS-docstring (p. 707f)."""
        # values from 1960-1978 with presample data
        B, sigma_u = models.VAR_LS(data, VAR_p)
        # sigma_u = np.cov(models.VAR_LS_residuals(data, B, VAR_p), ddof=1)
        npt.assert_almost_equal(B, B_test, decimal=3)
        npt.assert_almost_equal(sigma_u, sigma_u_test, decimal=6)

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
        npt.assert_almost_equal(np.corrcoef(data), np.corrcoef(sim), decimal=1)

    def test_VAR_LS_sim_m(self):
        """Does VAR_LS_sim change means as requested."""
        T = 500
        vg.reseed(0)
        sim = models.VAR_LS_sim(B_test, sigma_u_test, T)
        vg.reseed(0)
        m = np.array([2, 1, 0.5])
        mt = np.empty((len(m), T))
        mt.T[:] = m
        sim_m = models.VAR_LS_sim(B_test, sigma_u_test, T, m=mt)
        m_actual = sim_m.mean(axis=1) - sim.mean(axis=1)
        npt.assert_almost_equal(m_actual, m, decimal=1)

    def test_VAR_LS_sim_fixed_var(self):
        """Does VAR_LS_sim return the fixed values given via fixed_data?"""
        T = 5
        fixed = np.nan * np.empty(VAR_K * T).reshape((VAR_K, T))
        fixed[0] = np.array([0, 0, 1.0, 0, 0])
        sim = models.VAR_LS_sim(B_test, sigma_u_test, T, fixed_data=fixed)
        npt.assert_almost_equal(sim[0], fixed[0])

    def test_VARMA_LS_sim_fixed_var(self):
        """Does VAR_LS_sim return the fixed values given via fixed_data?"""
        T = 5
        fixed = np.nan * np.empty(VAR_K * T).reshape((VAR_K, T))
        fixed[0] = np.array([0, 0, 1.0, 0, 0])
        AM, sigma_u = models.VARMA_LS_prelim(data, VARMA_p, VARMA_q)[:-1]
        sim = models.VARMA_LS_sim(
            AM, VARMA_p, VARMA_q, sigma_u, data_means, T, fixed_data=fixed
        )
        npt.assert_almost_equal(sim[0], fixed[0])

    #  def test_VARMA_LS_sim(self):
    #      """Does VAR_LS_sim reproduce the correlation matrix?"""
    #      T = data.shape[1] * 100
    #      AM, sigma_u = models.VARMA_LS_prelim(data, VARMA_p, VARMA_q)[:-1]
    #      means = data.mean(axis=1)
    #      sim = models.VARMA_LS_sim(AM, VARMA_p, VARMA_q, sigma_u, means, T)
    #      npt.assert_almost_equal(np.corrcoef(data), np.corrcoef(sim), decimal=2)

    # def test_VARMA_LS_sim_VARMA_residuals(self):
    #     """Round-trip test: do we get predifined residuals back from
    #     VARMA_LS_residuals when simulating with VARMA_LS_sim?"""
    #     T = 5
    #     residuals_test = np.asmatrix(
    #                         [vg.rng.multivariate_normal(VAR_K * [0],
    #                                                        sigma_u_test)
    #                          for t in xrange(T)]).reshape((VAR_K, T))
    #     AM, sigma_u = models.VARMA_LS_prelim(data, VARMA_p, VARMA_q)[:-1]
    #     sim = models.VARMA_LS_sim(AM, VARMA_p, VARMA_q, sigma_u, T,
    #                               residuals_test, n_sim_multiple=1)
    #     residuals = models.VARMA_residuals(sim, AM, VARMA_p, VARMA_q)
    #     npt.assert_almost_equal(residuals, residuals_test, decimal=3)

    # def test_MGARCH_sim_MGARCH_residuals(self):
    #     vg.reseed(0)
    #     # B = [[-.24e-3, -.00, 0.02, -.18, .16, -.08, .11],
    #     #      [-.42e-3, 0.12, -.13, -.08, .03, -.01, .01]]
    #     # sigma_u = [[1., .8],
    #     #            [.8, 1.]]
    #     T = 1000
    #     sigma_z = .5
    #     a0, a1 = .25, .5
    #     sigma_t = np.ones((2, T))
    #     ut = np.zeros((2, T))
    #     zts = sigma_z * vg.rng.multivariate_normal([0, 0],
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
        residuals_test = np.array(
            [
                vg.rng.multivariate_normal(VAR_K * [0], sigma_u_test)
                for t in range(T)
            ]
        ).reshape((VAR_K, T))
        sim = models.VAR_LS_sim(
            B_test, sigma_u_test, T, u=residuals_test, n_presim_steps=0
        )
        residuals = models.VAR_residuals(sim, B_test, VAR_p)
        npt.assert_almost_equal(residuals, residuals_test)

    def test_VAR_LS_B_recover(self):
        T = 10000
        vg.reseed(0)
        sim = models.VAR_LS_sim(B_test, sigma_u_test, T)
        # fit on simulated values check if we can recover B
        B_fit, sigma_u_fit = models.VAR_LS(sim, p=VAR_p)
        try:
            npt.assert_almost_equal(B_fit, B_test, decimal=1)
            npt.assert_almost_equal(sigma_u_fit, sigma_u_test, decimal=4)
        except AssertionError:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
            axs[0, 0].matshow(B_test)
            axs[0, 0].set_title("B_test")
            axs[1, 0].matshow(B_fit)
            axs[1, 0].set_title("B_fit")
            axs[0, 1].matshow(sigma_u_test)
            axs[0, 1].set_title("sigma_u_test")
            axs[1, 1].matshow(sigma_u_fit)
            axs[1, 1].set_title("sigma_u_fit")
            plt.show()
            raise

    def test_VAREX_LS_sim_VAREX_residuals(self):
        """Round-trip test: do we get predifined residuals back from
        VAREX_LS_residuals when simulating with VAREX_LS_sim?"""
        T = 5
        residuals_test = np.array(
            [
                vg.rng.multivariate_normal(VAR_K * [0], sigma_u_test)
                for t in range(T)
            ]
        ).reshape((VAR_K, T))
        ex = np.full(T, 0.1)
        sim, ex_out = models.VAREX_LS_sim(
            Bex_test, sigma_u_test, T, ex, u=residuals_test, n_presim_steps=0
        )
        residuals = models.VAREX_residuals(sim, ex, Bex_test, VAR_p)
        npt.assert_almost_equal(residuals, residuals_test)
        npt.assert_almost_equal(ex_out, ex)

    def test_VAREX_LS_sim_VAREX_residuals_funcy(self):
        """Round-trip test: do we get predifined residuals back from
        VAREX_LS_residuals when simulating with VAREX_LS_sim?"""
        T = 5
        residuals_test = np.array(
            [
                vg.rng.multivariate_normal(VAR_K * [0], sigma_u_test)
                for t in range(T)
            ]
        ).reshape((VAR_K, T))
        ex_kwds = dict(fac=1.5)

        def ex(x, fac):
            return np.mean(x[:, -1]) * fac

        sim, ex_out = models.VAREX_LS_sim(
            Bex_test,
            sigma_u_test,
            T,
            ex,
            u=residuals_test,
            n_presim_steps=0,
            ex_kwds=ex_kwds,
        )
        residuals = models.VAREX_residuals(sim, ex, Bex_test, VAR_p, ex_kwds)
        npt.assert_almost_equal(residuals, residuals_test)

    def Bs_test(self, T):
        # have to have seasonal B_test and sigma_u_test
        Bs_test = np.empty(B_test.shape + (min(T, 365),))
        sin = 0.25 * np.sin(np.arange(min(T, 365)) * 2 * np.pi / 365)
        Bs_test[...] = sin[None, None, :] * np.asarray(B_test[..., None])
        Bs_test += B_test[..., None]
        # Bs_test[:, 0] = B_test[:, 0, None]
        return Bs_test

    def sigma_u_test_s(self, T):
        sigma_u_test_s = np.empty(
            (sigma_u_test.shape[0], sigma_u_test.shape[1], min(T, 365))
        )
        sin = 0.05 * np.sin(np.arange(min(T, 365)) * 2 * np.pi / 365)
        sigma_u_test_s[:] = sin[None, None, :] * np.asarray(
            sigma_u_test[..., None]
        )
        sigma_u_test_s += sigma_u_test[..., None]
        return sigma_u_test_s

    def test_sigma_u_test_s(self):
        """Is it symmetric and positive semi-definite?"""
        T = 365
        sigma_us = self.sigma_u_test_s(T)
        sym = [
            np.all((sigma_us[..., t] - sigma_us[..., t].T) == 0)
            for t in range(T)
        ]
        self.assertTrue(np.all(sym))
        for t in range(T):
            np.linalg.cholesky(sigma_us[..., t])

    def test_SVAR_LS_sim_SVAR_residuals(self):
        """Round-trip test: do we get predifined residuals back from
        SVAR_LS_residuals when simulating with SVAR_LS_sim?"""
        T = 5 * 365
        residuals_test = np.array(
            [
                vg.rng.multivariate_normal(VAR_K * [0], sigma_u_test)
                for t in range(T)
            ]
        ).reshape((VAR_K, T))
        doys = np.arange(1, T + 1) % 365
        # residuals_test[:] = 0
        Bs_test = self.Bs_test(T)
        models.SVAR_LS_sim.clear_cache()
        sim = models.SVAR_LS_sim(
            Bs_test,
            self.sigma_u_test_s(T),
            doys,
            u=residuals_test,
            n_presim_steps=0,
        )
        residuals = models.SVAR_residuals(sim, doys, Bs_test, VAR_p)
        residuals_test = np.asarray(residuals_test)
        # the beginning time steps will not be matched, but I do not care!
        try:
            npt.assert_almost_equal(
                residuals[:, VAR_p:], residuals_test[:, VAR_p:], decimal=6
            )
        except AssertionError:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(
                ncols=sim.shape[0], subplot_kw=dict(aspect="equal")
            )
            for var_i, ax in enumerate(axs):
                _ = ax.scatter(
                    residuals[var_i, VAR_p:], residuals_test[var_i, VAR_p:]
                )
            plt.show()
            raise

    def test_SVAR_LS_B_recover(self):
        T = 50 * 365
        vg.reseed(0)
        doys = np.arange(T) % 365
        Bs_test = self.Bs_test(T)
        sigma_u_test_s = self.sigma_u_test_s(T)
        B_test_mean = Bs_test.mean(axis=-1)
        sigma_u_test_mean = sigma_u_test_s.mean(axis=-1)
        K = Bs_test.shape[0]
        sim = models.SVAR_LS_sim(Bs_test, sigma_u_test_s, doys)
        # sigma_u_test_s[:, :] = sigma_u_test_mean[..., None]
        # sim = models.SVAR_LS_sim(Bs_test, sigma_u_test_s, doys)
        sim_stat = models.VAR_LS_sim(B_test_mean, sigma_u_test_mean, T)
        means = sim.mean(axis=1)
        means_stat = sim_stat.mean(axis=1)
        try:
            npt.assert_almost_equal(means, means_stat, decimal=2)
        except AssertionError:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(
                nrows=K, ncols=1, sharex=True, constrained_layout=True
            )
            means_th = np.array(
                [
                    np.squeeze(models.VAR_mean(Bs_test[..., t]))
                    for t in np.arange(T) % 365
                ]
            ).T
            means_stat_th = models.VAR_mean(B_test_mean)
            for var_i, ax in enumerate(axs):
                ax.plot(sim[var_i], "k", alpha=0.25)
                ax.plot(means_th[var_i], "k")
                ax.axhline(means[var_i], color="k")
                ax.plot(sim_stat[var_i], "b", alpha=0.25)
                ax.axhline(means_stat_th[var_i], color="b")
                ax.axhline(means_stat[var_i], color="b")
            plt.show()

        # std = sim.std(axis=1)
        # std_stat = sim_stat.std(axis=1)
        # npt.assert_almost_equal(std, std_stat, decimal=3)

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(nrows=2, ncols=K, figsize=(9, 6))
        ranks = (np.arange(T) - 0.5) / T
        for var_i in range(K):
            sim_sorted = np.sort(sim[var_i])
            sim_stat_sorted = np.sort(sim_stat[var_i])
            axs[0, var_i].plot(sim_sorted, ranks, label="seasonal")
            axs[0, var_i].plot(sim_stat_sorted, ranks, label="stationary")
            axs[1, var_i].plot(np.sort(sim_stat[var_i]), np.sort(sim[var_i]))
            smin = min(sim_sorted[0], sim_stat_sorted[0])
            smax = max(sim_sorted[-1], sim_stat_sorted[-1])
            axs[1, var_i].plot([smin, smax], [smin, smax], "--k")
            axs[1, var_i].set_xlabel("stationary")
        axs[1, 0].set_ylabel("seasonal")
        axs[0, 0].legend(loc="best")
        for ax in np.ravel(axs):
            ax.grid(True)
        plt.show()

        # fit on simulated values check if we can recover B and
        # sigma_u
        B_fit, sigma_u_fit = models.SVAR_LS(
            sim,
            doys,
            doy_width=20,
            fft_order=2,
            p=VAR_p,
        )
        try:
            npt.assert_almost_equal(B_fit, Bs_test, decimal=1)
            npt.assert_almost_equal(sigma_u_fit, sigma_u_test_s, decimal=4)
        except AssertionError:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(nrows=K, ncols=2, constrained_layout=True)
            J = Bs_test.shape[1]
            for k in range(K):
                for j in range(J):
                    t = (j - 1) // VAR_p
                    k_other = (j - 1) % VAR_p
                    axs[k, 0].plot(
                        Bs_test[k, j], label=f"(k={k_other}, t={t})"
                    )
                    axs[k, 0].axhline(B_test_mean[k, j])
                for j in range(K):
                    axs[k, 1].plot(
                        sigma_u_test_s[k, j], label=f"(k={k}, t={j})"
                    )
                    axs[k, 1].axhline(sigma_u_test_mean[k, j])
                for ax in axs[k]:
                    ax.set_prop_cycle(None)
                for j in range(J):
                    axs[k, 0].plot(B_fit[k, j], "--")
                for j in range(K):
                    axs[k, 1].plot(sigma_u_fit[k, j], "--")
                for ax in axs[k]:
                    ax.legend(loc="best")
            plt.show()
            raise


if __name__ == "__main__":
    npt.run_module_suite()
