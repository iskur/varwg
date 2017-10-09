import tempfile
import shutil
import numpy as np
import numpy.testing as npt
from vg.time_series_analysis import time_series
import vg
from vg.time_series_analysis import cresample
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()},
                  reload_support=True)


class Test(npt.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.tempdir = tempfile.mkdtemp()
        self.met_vg = vg.VG((  # "R",
            "theta", "Qsw", "ILWR", "rh", "u", "v"), verbose=False)
        self.data = self.met_vg.data_trans
        self.times = self.met_vg.times

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_resample(self):
        res = cresample.resample(self.data, self.times, p=3)[0]
        npt.assert_almost_equal(time_series.auto_corr(self.data, 0),
                                time_series.auto_corr(res, 0))

    def test_resample_mean(self):
        np.random.seed(0)
        res = cresample.resample(self.data, self.times, p=3,
                                 theta_incr=0)[0]
        npt.assert_almost_equal(np.mean(res, axis=1),
                                np.mean(self.data, axis=1),
                                decimal=1)
        theta_incr = 1.

        # we only expect to hit the right theta_incr in the mean of a
        # lot of realizations.  so we do a very long simulation which
        # is about the same thing
        # n_sim_steps = 10 * self.data.shape[1]
        n_sim_steps = None
        theta_i = self.met_vg.var_names.index("theta")
        means = []
        data_mean = np.mean(self.data[theta_i])
        theta_incrs = np.linspace(0, .8, 15, endpoint=False)
        for theta_incr in theta_incrs:
            ress = []
            for _ in range(2):
                res = cresample.resample(self.data, self.times, p=3,
                                         n_sim_steps=n_sim_steps,
                                         theta_incr=theta_incr,
                                         theta_i=theta_i,
                                         cache_dir=self.tempdir,
                                         verbose=False)[0]
                ress += [res[theta_i]]
            means += [np.mean(ress) - data_mean]

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        # ax.scatter(theta_incrs, means)
        # min_ = np.min(np.array([means, theta_incrs]))
        # max_ = np.max(np.array([means, theta_incrs]))
        # ax.plot([min_, max_], [min_, max_])
        # ax.set_xlabel("expected means")
        # ax.set_ylabel("actual means")
        # ax.grid(True)
        # plt.show()

        npt.assert_almost_equal(means, theta_incrs, decimal=1)
