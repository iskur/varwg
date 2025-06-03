import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import numpy as np
import numpy.testing as npt

import vg

config_template = vg.config_template
vg.set_conf(config_template)

seed = 0
p = 3
T = 3 * 365
fit_kwds = dict(p=p, fft_order=3, doy_width=15, seasonal=True)
var_names = (
    # we do not use precipitation here as long as we cannot
    # disaggregate it properly
    "R",
    "theta",
    "Qsw",
    "ILWR",
    "rh",
    "u",
    "v",
)
script_home = os.path.dirname(vg.__file__)
met_file = os.path.join(script_home, "sample.met")
data_dir = os.path.join(os.path.dirname(vg.core.__file__), "tests", "data")
sim_file = os.path.join(data_dir, "test_out_sample.met")

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(met_file):
    # try to retrieve it from the VG repository on bitbucket
    import urllib.request

    url = "https://bitbucket.org/iskur/vg/raw/tip/vg/sample.met"
    print("Downloading sample data for VG.")
    urllib.request.urlretrieve(url, met_file)

if not os.path.exists(sim_file):
    from vg.core.tests import gen_test_data

    gen_test_data.main()


class TestVG(npt.TestCase):
    def setUp(self):
        self.verbose = False
        met = vg.read_met(sim_file, verbose=self.verbose)[1]
        self.sim = np.squeeze(
            np.array([[met[var_name]] for var_name in var_names])
        )
        self.data_dir = tempfile.mkdtemp("vg_temporary_test")
        self.met_vg = vg.VG(
            var_names,
            refit="all",
            data_dir=self.data_dir,
            cache_dir=self.data_dir,
            met_file=met_file,
            verbose=self.verbose,
        )
        vg.reseed(seed)
        self.met_vg.fit(p)

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    @npt.decorators.slow
    def test_plain(self):
        self.assertEqual(self.met_vg.p, p)
        np.random.seed(1)
        self.met_vg.simulate(T=T)
        sim = self.met_vg.disaggregate()[1]
        assert np.all(np.isfinite(sim))
        npt.assert_almost_equal(sim, self.sim, decimal=2)

    @npt.decorators.slow
    def test_seasonal(self):
        """Test only for exceptions, not for equality of results."""
        self.met_vg.fit(p, seasonal=True)
        self.met_vg.simulate(T)
        del self.met_vg.Bs, self.met_vg.sigma_us, self.met_vg.seasonal

    # @npt.decorators.slow
    # def test_extro(self):
    #     import matplotlib.pyplot as plt
    #     self.met_vg.simulate()
    #     self.met_vg.plot_autocorr()
    #     self.met_vg.plot_VAR_par()
    #     print(np.round(self.met_vg.AM, 2))

    #     self.met_vg.fit(extro=True, p=p)
    #     self.met_vg.plot_VAR_par()
    #     print(np.round(self.met_vg.AM, 2))
    #     self.met_vg.simulate()
    #     self.met_vg.plot_autocorr()
    #     plt.show()

    @npt.decorators.slow
    def test_multi_prim(self):
        # testing a change in humidity together with an increase in theta
        rh_signal, _ = self.met_vg.random_dryness(
            duration_min=7, duration_max=14
        )
        self.met_vg.simulate(
            primary_var=("theta", "rh"),
            climate_signal=(None, rh_signal),
            theta_incr=(4, None),
        )

    # @npt.decorators.slow
    # def test_rr_fact(self):
    #     r_fact = 1.5
    #     met_vg = vg.VG(("theta", "R", "Qsw"), data_dir=self.data_dir,
    #                    cache_dir=self.data_dir, met_file=met_file,
    #                    verbose=False)
    #     times, sim = met_vg.simulate(r_fact=r_fact)
    #     r_index = met_vg.var_names.index("R")
    #     npt.assert_almost_equal(np.nanmean(sim[r_index]),
    #                             np.nanmean(r_fact * met_vg.data_raw[r_index] /
    #                                        met_vg.sum_interval[r_index]))

    @npt.decorators.slow
    def test_resample(self):
        # use the constance data set here.  the self-generated data is
        # strange.... (hint: look what you have done there!)
        import config_konstanz as conf

        vg.core.vg_core.conf = vg.conf = vg.vg_base.conf = conf
        met_vg = vg.VG(
            ("theta", "ILWR", "Qsw", "rh", "u", "v"),
            refit=True,
            verbose=self.verbose,
        )
        met_vg.fit(p=3)
        # shelve_filepath = os.path.join(vg.conf.cache_dir,
        #                                vg.resampler.shelve_filename)
        # if os.path.exists(shelve_filepath):
        #     os.remove(shelve_filepath)
        theta_incr = 4
        mean_arrival = 7
        disturbance_std = 5
        # kwds = dict(start_str="01.01.1994 00:00:00",
        #             stop_str="02.01.2015 00:00:00")
        kwds = dict()
        res_dict_nocy = vg.my.ADict(recalibrate=True, n_candidates=20)
        res_dict_cy = res_dict_nocy + dict(cy=True)
        theta_mean = np.mean(met_vg.data_raw[0] / met_vg.sum_interval)
        for res_dict in (res_dict_nocy, res_dict_cy):
            kwds["res_kwds"] = res_dict
            # met_vg.simulate(**kwds)
            simt, sim = met_vg.simulate(theta_incr=theta_incr, **kwds)
            sim_diff = np.mean(sim[0]) - theta_mean
            npt.assert_almost_equal(sim_diff, theta_incr, decimal=0)
            simt, sim = met_vg.simulate(
                mean_arrival=mean_arrival,
                disturbance_std=disturbance_std,
                **kwds
            )
            sim_diff = np.mean(sim[0]) - theta_mean
            npt.assert_almost_equal(sim_diff, 0, decimal=0)
            # simt, sim = met_vg.simulate(theta_incr=theta_incr,
            #                             mean_arrival=mean_arrival,
            #                             disturbance_std=disturbance_std,
            #                             **kwds)
            # sim_diff = np.mean(sim[0]) - theta_mean
            # npt.assert_almost_equal(sim_diff, theta_incr, decimal=0)

    @npt.decorators.slow
    def test_theta_incr(self):
        import config_konstanz

        vg.core.vg_core.conf = vg.conf = vg.vg_base.conf = config_konstanz
        met_vg = vg.VG(
            ("theta", "ILWR", "Qsw", "rh", "u", "v"), verbose=self.verbose
        )
        met_vg.fit(p=3, seasonal=True)
        theta_incr = 4
        mean_arrival = 7
        disturbance_std = 5
        data_mean = np.mean(met_vg.data_raw[0]) / 24.0
        simt, sim = met_vg.simulate(theta_incr=theta_incr)
        npt.assert_almost_equal(
            sim[0].mean() - data_mean, theta_incr, decimal=1
        )
        simt, sim = met_vg.simulate(
            theta_incr=theta_incr,
            mean_arrival=mean_arrival,
            disturbance_std=disturbance_std,
        )
        npt.assert_almost_equal(
            sim[0].mean() - data_mean, theta_incr, decimal=1
        )

    @npt.decorators.slow
    def test_rainmix(self):
        import matplotlib.pyplot as plt
        import config_konstanz_disag as conf
        from vg import vg_plotting

        vg.conf = vg.vg_base.conf = vg.vg_plotting.conf = conf
        met_vg = vg.VG(
            ("R", "theta", "ILWR", "Qsw", "rh", "u", "v"),
            # non_rain=("theta", "Qsw", "rh"),
            # refit="R",
            # refit=True,
            verbose=self.verbose,
            dump_data=True,
        )
        met_vg.fit(3)
        simt, sim = met_vg.simulate()
        # simt_dis, sim_dis = met_vg.disaggregate()
        # met_vg.plot_meteogramm()
        # plt.show()

    # def test_rm(self):
    #     import config_konstanz_disag
    #     vg.conf = vg.vg_base.conf = config_konstanz_disag
    #     met_vg = vg.VG(("theta", "ILWR", "Qsw", "rh", "u", "v"),
    #                    verbose=True)
    #     met_vg.simulate()
    #     met_vg.disaggregate_rm()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    import warnings

    warnings.simplefilter("error", RuntimeWarning)
    if os.path.exists(met_file):
        npt.run_module_suite()
