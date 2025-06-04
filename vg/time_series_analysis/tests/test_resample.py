import tempfile
import shutil
import numpy as np
import numpy.testing as npt
from vg.time_series_analysis import resample, time_series
import vg

# import config_konstanz
# vg.conf = vg.vg_base.conf = config_konstanz


class Test(npt.TestCase):
    def setUp(self):
        vg.reseed(0)
        self.tempdir = tempfile.mkdtemp()
        self.met_vg = vg.VG(
            ("theta", "Qsw", "ILWR", "rh", "u", "v"), verbose=False  # "R",
        )
        self.data = self.met_vg.data_trans
        self.times = self.met_vg.times
        self.verbose = False

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_resample(self):
        res = resample.resample(self.data, self.times, p=3)[0]
        npt.assert_almost_equal(
            time_series.auto_corr(self.data, 0), time_series.auto_corr(res, 0)
        )

    def test_bias_funcs(self):
        a, b, c, d = 1.245, 0.978, 1.205, 0.120
        theta_incr = 4
        mean = resample.bias2mean(4, a, b, c, d)
        theta_incr_est = resample.mean2bias(mean, a, b, c, d)
        npt.assert_almost_equal(theta_incr_est, theta_incr)

    def test_resample_mean(self):
        vg.reseed(0)
        res = resample.resample(self.data, self.times, p=3, theta_incr=0)[0]
        npt.assert_almost_equal(
            np.mean(res, axis=1), np.mean(self.data, axis=1), decimal=1
        )
        theta_incr = 1.5

        # we only expect to hit the right theta_incr in the mean of a
        # lot of realizations.  so we do a very long simulation which
        # is about the same thing
        n_sim_steps = 10 * self.data.shape[1]
        # n_sim_steps = None
        theta_i = self.met_vg.var_names.index("theta")
        means = []
        data_mean = np.mean(self.data[theta_i])
        theta_incrs = np.linspace(0, 0.8, 15, endpoint=False)
        for theta_incr in theta_incrs:
            ress = []
            for _ in range(2):
                res = resample.resample(
                    self.data,
                    self.times,
                    p=3,
                    n_sim_steps=n_sim_steps,
                    theta_incr=theta_incr,
                    theta_i=theta_i,
                    cache_dir=self.tempdir,
                    verbose=False,
                )[0]
                ress += [res[theta_i]]
            means += [np.mean(ress) - data_mean]

        try:
            npt.assert_almost_equal(means, theta_incrs, decimal=1)
        except AssertionError:
            if not self.verbose:
                raise
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(
                nrows=1, ncols=1, subplot_kw=dict(aspect="equal")
            )
            ax.scatter(theta_incrs, means, marker="x")
            ax.set_xlabel("expected mean increases")
            ax.set_ylabel("actual mean increases")
            ax.grid(True)
            ax.axline([0, 0], slope=1, color="k", linestyle="--")
            plt.show()
            raise
