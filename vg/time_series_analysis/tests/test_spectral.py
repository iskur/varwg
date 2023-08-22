import numpy as np
import numpy.testing as npt
from vg.time_series_analysis import spectral, time_series, models
from vg.time_series_analysis.tests import test_models
import vg

# these are empirical covariances from stdn constance data
# theta, Qsw and ILWR
# cov = np.array([[0.92887129,  0.23603728,  0.58140583],
#                 [0.23603728,  1.01204546, -0.44856305],
#                 [0.58140583, -0.44856305,  0.9826686]])
cov = np.array([[0.93, 0.24, 0.58], [0.24, 1.01, -0.45], [0.58, -0.45, 0.98]])
autocov_data = np.array(
    [
        [0.93, 0.75, 0.54, 0.41, 0.32, 0.25, 0.2, 0.17],
        [1.01, 0.31, 0.12, 0.06, 0.04, 0.02, 0.01, 0.03],
        [0.98, 0.62, 0.34, 0.22, 0.16, 0.11, 0.09, 0.08],
    ]
)
K = cov.shape[0]


def ac(var_i):
    """An autocovariance "model" based on the empirical autocovariances
    above."""

    def ac_(lag):
        lag = np.atleast_1d(lag)
        ii = lag < autocov_data.shape[1]
        covs = np.zeros_like(lag) + 1e-6
        covs[ii] = autocov_data[var_i, lag[ii]]
        return np.squeeze(covs)

    return ac_


autocovs = [ac(0), ac(1), ac(2)]


class Test(npt.TestCase):
    def setUp(self):
        vg.reseed(3)
        # spectral.simulate is sensitive towards T! low T -> bad fit
        self.T = 25000
        # generating a ar-1 time series which has a well known
        # autocovariance function
        # high rho, worse fit
        self.rho = rho = 0.75
        test_data = vg.rng.normal(size=self.T)
        for t in range(1, self.T):
            test_data[t] += rho * test_data[t - 1]
        self.test_data = test_data - test_data.mean()
        self.test_data /= self.test_data.std()

    def tearDown(self):
        pass

    def test_1d_simulate(self):
        def cov_model_ar1(lag):
            return self.rho**lag / (1 - self.rho**2)

        n_lags = 10
        cov_exp = cov_model_ar1(np.arange(n_lags))
        spec = spectral.Spectral(cov_model_ar1, self.T, pool_size=2)
        data_sim = spec.sim_n(5)
        for data in data_sim:
            cov_act = np.array(
                [time_series.auto_cov(data, lag) for lag in range(n_lags)]
            )
            npt.assert_almost_equal(cov_act, cov_exp, decimal=1)
        data_sim = spec.sim
        cov_act = np.array(
            [time_series.auto_cov(data_sim, lag) for lag in range(n_lags)]
        )
        npt.assert_almost_equal(cov_act, cov_exp, decimal=1)

    def test_nd_simulate(self):
        cov_model = lambda h: np.exp(-h)
        domainshape = 500, 500, 10
        # cov_exp = cov_model(np.arange(domainshape[0]))
        spec = spectral.SpectralND(
            cov_model, domainshape, pool_size=2, scale=(10, 5, 1)
        )
        # cov_act =
        data_sim = spec.sim
        npt.assert_array_equal(data_sim.shape, domainshape)
        npt.assert_almost_equal(data_sim.std(), 1.0, decimal=1)
        # can we get a different standard deviation?
        spec.sigma = sigma = 2.5
        npt.assert_almost_equal(spec.sim.std(), spec.sigma, decimal=1)
        spec = spectral.SpectralND(
            cov_model, domainshape, pool_size=2, scale=(10, 5, 1), sigma=sigma
        )
        data_sims = spec.sim_n(5)
        for data_sim in data_sims:
            npt.assert_array_equal(data_sim.shape, domainshape)
            npt.assert_almost_equal(data_sim.std(), spec.sigma, decimal=1)

    def test_2d_simulate(self):
        # def cov_model_ar1(rho):
        #     return lambda lag: rho ** lag / (1 - rho ** 2)
        # these are mild differences in autocorrelation. with more
        # different ones the last test fails
        # autocovs = [cov_model_ar1(.6),
        #             cov_model_ar1(.65),
        #             cov_model_ar1(.75)]
        mspec = spectral.MultiSpectral(autocovs, cov, self.T)
        data_sim = mspec.sim
        cov_sim = np.cov(data_sim)
        npt.assert_almost_equal(cov_sim, cov, decimal=2)
        lags = list(range(1, 7))
        for var_i, var in enumerate(data_sim):
            cov0 = autocovs[var_i](0)
            ac_exp = [autocovs[var_i](lag) / cov0 for lag in lags]
            ac_act = [time_series.auto_corr(var, lag) for lag in lags]
            npt.assert_almost_equal(ac_act, ac_exp, decimal=1)

    def test_spectral_2d_covdata(self):
        """Test if MultiSpectral works with (auto)covs given as array."""
        T = 7500
        data_obs = models.VAR_LS_sim(
            test_models.B_test, test_models.sigma_u_test, T
        )
        data_obs -= data_obs.mean(axis=1)[:, None]
        data_obs /= data_obs.std(axis=1)[:, None]
        mspec = spectral.MultiSpectral(data_obs, data_obs, T)
        data_sim = mspec.sim
        npt.assert_almost_equal(np.cov(data_sim), np.cov(data_obs), decimal=1)
        for var_i in range(data_obs.shape[0]):
            ac_obs = [
                time_series.auto_cov(data_obs[var_i], lag) for lag in range(7)
            ]
            ac_sim = [
                time_series.auto_cov(data_sim[var_i], lag) for lag in range(7)
            ]
            npt.assert_almost_equal(ac_sim, ac_obs, decimal=1)


if __name__ == "__main__":
    npt.run_module_suite()
