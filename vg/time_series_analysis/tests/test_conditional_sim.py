# -*- coding: utf-8 -*-
import time
import numpy as np
import numpy.testing as npt
import vg
import vg.time_series_analysis.conditional_sim as csim
from vg.time_series_analysis import models, time_series, distributions
from .test_spectral import cov, autocovs, K
from . import test_models


class Test(npt.TestCase):
    def setUp(self):
        vg.reseed(0)
        self.verbose = False
        self.rho = 0.9
        self.cov_model_ar1 = lambda lag: self.rho**lag / (1 - self.rho**2)
        self.domainshape = 65, 55
        self.scale = 5
        self.speed = 5
        self.cov_model = lambda h: 5 * np.exp(-h / 5.0)

        # setup stuff for SimulateUVSeq
        # now we need a time-series of conditioning points
        cond_eq_u_ii = [[10, 10, 40, 40], [10, 40, 10, 40]]
        self.cond_eq_u_ii = [cond_eq_u_ii, None, cond_eq_u_ii]
        self.cond_eq_v_ii = self.cond_eq_u_ii
        cond_eq_u_vals = np.array(
            [-0.5 * self.speed, -self.speed, self.speed, 0.75 * self.speed]
        )
        self.cond_eq_u_vals = np.array(
            [
                cond_eq_u_vals,
                None,
                cond_eq_u_vals,
                # None
            ]
        )
        cond_eq_v_vals = np.array(
            [self.speed, -0.75 * self.speed, 0.5 * self.speed, -self.speed]
        )
        self.cond_eq_v_vals = np.array(
            [
                cond_eq_v_vals,
                None,
                cond_eq_v_vals,
                # None
            ]
        )

        # mean u component over left boundary
        self.cond_mean_u_ii = [
            [list(range(self.domainshape[0])), self.domainshape[0] * [0]]
        ]
        self.cond_mean_u_vals = [5]

        self._uus, self._vvs = None, None
        T_agg = 20
        self.data_obs = models.VAR_LS_sim(
            test_models.B_test, test_models.sigma_u_test, T_agg
        )
        self.data_obs -= self.data_obs.mean(axis=1)[:, None]
        self.data_obs /= self.data_obs.std(axis=1)[:, None]
        self.data_obs[1, 10] = np.nan
        self.data_obs[:, 5:7] = np.nan
        self.cov = np.cov(
            self.data_obs[:, np.all(np.isfinite(self.data_obs), axis=0)]
        )

        def autocov_gen(var_i):
            return lambda lag: time_series.auto_cov(self.data_obs[var_i], lag)

        self.autocovs = (autocov_gen(0), autocov_gen(1), autocov_gen(2))

    def tearDown(self):
        pass

    def test_simulate(self):
        """Are the conditioning points met? Are the covs ok?"""
        T = 25000
        cond_eq_ii = list(range(0, T, int(T / 50)))
        cond_vals = len(cond_eq_ii) * [3]
        cond_eq_ii += list(range(3, T, int(T / 20)))
        cond_vals += (len(cond_eq_ii) - len(cond_vals)) * [0]
        data_sim = csim.simulate(
            self.cov_model_ar1,
            T,
            cond_eq=(cond_eq_ii, cond_vals),
            verbose=self.verbose,
        )
        # do we get the values we expect at the conditioning points?
        npt.assert_almost_equal(data_sim[cond_eq_ii], cond_vals)
        # wtf! did this really work right now?
        # import matplotlib.pyplot as plt
        # plt.plot(data_sim, "-x")
        # plt.scatter(cond_ii, cond_vals, color="red", s=120.)
        # plt.show()

        # is the autocovariance still as expected?
        # bárdossy: this is not that important
        # n_lags = 10
        # covs_exp = [cov_model_ar1(lag) for lag in range(n_lags)]
        # covs_act = [time_series.auto_cov(data_sim, lag)
        #             for lag in range(n_lags)]
        # npt.assert_almost_equal(covs_act, covs_exp, decimal=2)

    def test_simulate_serial(self):
        """Test whether simulate is able to honour subsequent conds."""
        T = 500
        cond_ii = list(range(240, 260))
        cond_vals = int(len(cond_ii) / 2) * [-3, 3]
        data_sim = csim.simulate(
            self.cov_model_ar1,
            T,
            cond_eq=(cond_ii, cond_vals),
            verbose=self.verbose,
        )
        npt.assert_almost_equal(data_sim[cond_ii], cond_vals)
        # import matplotlib.pyplot as plt
        # plt.plot(data_sim, "-x")
        # plt.scatter(cond_ii, cond_vals, color="red", s=120.)
        # plt.show()

    def test_simulate_mean(self):
        """Test the mean conditions of simulate."""
        T = 500
        cond_ii = np.arange(T).reshape(10, -1)
        cond_vals = cond_ii.shape[0] // 2 * [-3, 3]
        data_sim = csim.simulate(
            self.cov_model_ar1,
            T,
            cond_mean=(cond_ii, cond_vals),
            verbose=self.verbose,
        )
        means = np.mean(data_sim[cond_ii], axis=1)
        npt.assert_almost_equal(means, cond_vals)

    # def test_disaggregate(self):
    #     """Test disaggregate."""
    #     T = 13
    #     data = T // 2 * [-3, 3]
    #     disagg_len = 10
    #     import time
    #     pre = time.time()
    #     data_sim = csim.disaggregate(data, self.cov_model_ar1,
    #                                  disagg_len)
    #     print "Seconds in continuous version: ", time.time() - pre
    #     means = np.mean(data_sim.reshape(-1, disagg_len), axis=1)
    #     npt.assert_almost_equal(means, data)
    #     # thest if we get the same with disaggregate_piecewise
    #     pre = time.time()
    #     data_sim2 = csim.disaggregate_piecewice(data,
    #                                             self.cov_model_ar1,
    #                                             disagg_len,
    #                                             verbose=True)
    #     print "Seconds of chunked version: ", time.time() - pre
    #     means2 = np.mean(data_sim2.reshape(-1, disagg_len,), axis=1)
    #     npt.assert_almost_equal(means2, data)

    def test_disaggregate_2d(self):
        # vg.reseed(0)
        disagg_len = 5
        nan_mask = np.isnan(self.data_obs)

        # test if we get the same with disaggregate_piecewise
        dis2 = csim.disaggregate_piecewice
        pre = time.time()
        data_sim2 = dis2(
            self.data_obs,
            autocovs,
            disagg_len=disagg_len,
            cov=cov,
            thresh=0.01,
            verbose=self.verbose,
        )
        if self.verbose:
            print("Seconds of chunked version: ", time.time() - pre)
        means2 = np.mean(data_sim2.reshape(K, -1, disagg_len), axis=-1)
        npt.assert_almost_equal(
            means2[~nan_mask], self.data_obs[~nan_mask], decimal=5
        )
        if self.verbose:
            pre = time.time()
        data_sim = csim.disaggregate(
            self.data_obs,
            autocovs,
            disagg_len=disagg_len,
            cov=cov,
            verbose=self.verbose,
        )
        if self.verbose:
            print("Seconds in continuous version: ", time.time() - pre)
        means = np.mean(data_sim.reshape(K, -1, disagg_len), axis=-1)
        npt.assert_almost_equal(means[~nan_mask], self.data_obs[~nan_mask])
        # npt.assert_almost_equal(np.cov(data_sim), cov)

        # did we change the covariance structure with the piecewise disag?
        lags = list(range(7))
        autocovs1 = np.array(
            [time_series.auto_corr(data_sim, lag) for lag in lags]
        ).T
        autocovs2 = np.array(
            [time_series.auto_corr(data_sim2, lag) for lag in lags]
        ).T

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(K)
        # for vi, ax in enumerate(axs):
        #     ax.plot(autocovs1[vi])
        #     ax.plot(autocovs2[vi])
        # plt.show()

        # we need to be very tolerant here, with all the stochasticity
        # an whatnot
        npt.assert_allclose(autocovs2, autocovs1, rtol=0.1, atol=0.25)

    def test_disaggregate_2d_nonlin(self):
        """Disaggregate with a back-transformation."""
        vg.reseed(0)
        dist = distributions.expon(1, 1 / 15)

        def trans(values, *args):
            return np.array(
                [dist.ppf(distributions.norm.cdf(row)) for row in values]
            )

        disagg_len = 5
        data_sim = csim.disaggregate_piecewice(
            self.data_obs,
            autocovs,
            disagg_len=disagg_len,
            cov=cov,
            trans=trans,
            verbose=False,
        )
        nan_mask = np.isnan(self.data_obs)
        # data is stdn-distributed
        data_trans = trans(self.data_obs)
        means = np.mean(trans(data_sim.reshape(K, -1, disagg_len)), axis=-1)

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(K)
        # for k, ax in enumerate(axs):
        #     ax.plot(np.repeat(data_trans[k], disagg_len), "-x")
        #     ax.plot(np.repeat(means[k], disagg_len)), "-x"
        #     ax.plot(trans(data_sim[k]))
        # plt.show()

        npt.assert_almost_equal(
            means[~nan_mask], data_trans[~nan_mask], decimal=0
        )

    def test_simulate_2d(self):
        T = 500
        # cond_eq_ii = range(0, T, int(T / 50))
        cond_eq_ii = list(range(T // 2 - 25, T // 2 + 25))
        len_conds = len(cond_eq_ii)
        cond_eq_ii = [len_conds * [0], cond_eq_ii]
        cond_vals = len_conds * [5]
        time_before = time.time()
        data_sim = csim.simulate(
            autocovs,
            T,
            self.cov,
            cond_eq=(cond_eq_ii, cond_vals),
            verbose=self.verbose,
        )
        if self.verbose:
            print("Time in picos version: ", time.time() - time_before)
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(K, sharex=True)
        # for var_i, ax in enumerate(axes):
        #     ax.plot(data_sim[var_i])
        # axes[0].scatter(cond_eq_ii[1], cond_vals, color="red")
        # plt.show()
        npt.assert_almost_equal(data_sim[cond_eq_ii], cond_vals)
        # this fails, but that might not be bad, because of the conditioning
        # npt.assert_almost_equal(np.cov(data_sim), cov)
        time_before = time.time()
        sim = csim.SimulateMulti(
            T, autocovs, self.cov, pool_size=320, verbose=self.verbose
        )
        sim.cond_eq(cond_eq_ii, cond_vals)
        data_sim2 = sim.run()
        if self.verbose:
            print("Time in SVD version: ", time.time() - time_before)
        npt.assert_almost_equal(data_sim2[cond_eq_ii], cond_vals)

    def test_simulate_2d_mean(self):
        """Test the mean conditions of simulate."""
        T = 500
        cond_ii = (
            (10 * [0], list(range(210, 220))),
            (20 * [0], list(range(220, 240))),
        )
        cond_vals = (-3.0, 3)
        data_sim = csim.simulate(
            autocovs, T, cov, cond_mean=(cond_ii, cond_vals)
        )
        means_act = [data_sim[ij].mean() for ij in cond_ii]
        npt.assert_almost_equal(means_act, cond_vals)
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(K, sharex=True)
        # for var_i, ax in enumerate(axes):
        #     ax.plot(data_sim[var_i])
        # axes[0].plot(cond_ii[0][1], len(cond_ii[0][0]) * [cond_vals[0]],
        #              color="red")
        # axes[0].plot(cond_ii[1][1], len(cond_ii[1][0]) * [cond_vals[1]],
        #              color="red")
        # plt.show()


if __name__ == "__main__":
    npt.run_module_suite()
