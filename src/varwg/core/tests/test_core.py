from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

import varwg
from varwg import helpers as my

config_template = varwg.config_template
varwg.set_conf(config_template)

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
disagg_varnames = [name for name in var_names if name != "R"]
script_home = Path(varwg.__file__).parent
met_file = script_home / "sample.met"
data_dir = Path(varwg.core.__file__).parent / "tests" / "data"
sim_file = data_dir / "test_out_sample.met"

data_dir.mkdir(exist_ok=True)

# Ensure sample.met file exists (should be bundled with package)
if not met_file.exists():
    raise FileNotFoundError(
        f"Sample data file not found at {met_file}. "
        "This file should be bundled with the package."
    )

if not sim_file.exists():
    from . import gen_test_data

    gen_test_data.main()


class TestVarWG(npt.TestCase):
    """VarWG test suite with class-level caching for performance.

    Uses class-level cached VarWG instances that are created once and reused
    across all tests, instead of being recreated in setUp() for every test.
    This reduces test time from ~26 minutes to much faster.
    """

    # Class-level cache for expensive VarWG instances
    _cached_varwg_regr = None
    _cached_varwg_dist = None
    _cached_varwg_sim = None
    _cached_sample_sim = None
    _cached_data_dir = None

    verbose = False
    refit = True

    def setUp(self):
        """Set up test fixtures, using class-level cache when available."""
        # Use class-level cached instances if available
        if TestVarWG._cached_varwg_regr is not None:
            self.met_varwg = TestVarWG._cached_varwg_regr
            self.varwg_regr = TestVarWG._cached_varwg_regr
            self.varwg_dist = TestVarWG._cached_varwg_dist
            self.varwg_sim = TestVarWG._cached_varwg_sim
            self.sample_sim = TestVarWG._cached_sample_sim
            self.data_dir = TestVarWG._cached_data_dir
            return

        # First time: create and cache expensive VarWG instances
        met = varwg.read_met(
            sim_file, verbose=self.verbose, with_conversions=True
        )[1]
        self.sample_sim = np.array([met[var_name] for var_name in var_names])
        self.data_dir = tempfile.mkdtemp("varwg_temporary_test")
        kwds = dict(
            refit=self.refit,
            data_dir=self.data_dir,
            cache_dir=self.data_dir,
            met_file=met_file,
            verbose=self.verbose,
            infill=True,
            station_name="test",
        )

        varwg.reseed(seed)
        self.met_varwg = varwg.VarWG(
            var_names, rain_method="regression", **kwds
        )
        self.varwg_regr = self.met_varwg
        # Fit all instances (expensive operation, done only once)
        self.varwg_dist = varwg.VarWG(
            var_names, rain_method="distance", **kwds
        )
        self.varwg_sim = varwg.VarWG(
            var_names, rain_method="simulation", **kwds
        )
        self.met_varwg.fit(**fit_kwds)
        self.varwg_dist.fit(**fit_kwds)
        self.varwg_sim.fit(**fit_kwds)

        # Cache for future tests
        TestVarWG._cached_varwg_regr = self.met_varwg
        TestVarWG._cached_varwg_dist = self.varwg_dist
        TestVarWG._cached_varwg_sim = self.varwg_sim
        TestVarWG._cached_sample_sim = self.sample_sim
        TestVarWG._cached_data_dir = self.data_dir

    # def tearDown(self):
    #     shutil.rmtree(self.data_dir)

    def test_diff(self):
        assert not self.varwg_dist._diff(self.varwg_dist)
        diff = self.varwg_dist._diff(
            self.varwg_regr, verbose=False, plot=False
        )
        # from varwg import helpers as my

        # print(my.key_tree(diff))
        assert "rain_method" in diff
        assert diff["rain_method"] == ("distance", "regression")

    def test_serialization(self):
        # if self.refit:
        #     met_varwg_fresh = self.met_varwg
        # else:
        #     met_varwg_fresh = varwg.VarWG(
        #         var_names,
        #         rain_method="regression",
        #         refit=True,
        #         data_dir=self.data_dir,
        #         cache_dir=self.data_dir,
        #         met_file=met_file,
        #         verbose=self.verbose,
        #     )
        #     varwg.reseed(seed)
        met_varwg_fresh = varwg.VarWG(
            var_names,
            rain_method="simulation",
            refit=True,
            data_dir=self.data_dir,
            cache_dir=self.data_dir,
            met_file=met_file,
            verbose=self.verbose,
        )
        varwg.reseed(seed)
        met_varwg_pickle = varwg.VarWG(
            var_names,
            rain_method="simulation",
            refit=False,
            # refit=True,
            data_dir=self.data_dir,
            cache_dir=self.data_dir,
            met_file=met_file,
            verbose=self.verbose,
        )
        assert not met_varwg_fresh._diff(
            met_varwg_pickle, verbose=2, plot=True
        )

        def non_rain(data):
            return np.array(
                list(
                    data[var_i]
                    for var_i, var_name in enumerate(var_names)
                    if var_name != "R"
                )
            )

        non_rain_fresh = non_rain(met_varwg_fresh.data_trans)
        non_rain_pickle = non_rain(met_varwg_pickle.data_trans)

        try:
            npt.assert_almost_equal(non_rain_fresh, non_rain_pickle)
        except AssertionError as exc:
            if self.verbose:
                print(met_varwg_pickle._diff(met_varwg_fresh))
                fig, axs = plt.subplots(
                    nrows=len(var_names) - 1, ncols=1, sharex=True
                )
                for var_i, var_name in enumerate(var_names[1:]):
                    ax = axs[var_i]
                    ax.plot(
                        met_varwg_fresh.times,
                        non_rain_fresh[var_i],
                        label="fresh",
                    )
                    ax.plot(
                        met_varwg_pickle.times,
                        non_rain_pickle[var_i],
                        label="pickle",
                    )
                    ax.set_title(var_name)
                    ax.grid(True)
                plt.show()
                raise exc

    def test_negative_rain(self):
        rain_mask_regr = self.varwg_regr.rain_mask
        rain_mask_dist = self.varwg_dist.rain_mask
        rain_mask_sim = self.varwg_sim.rain_mask
        npt.assert_almost_equal(rain_mask_regr, rain_mask_dist)
        npt.assert_almost_equal(rain_mask_regr, rain_mask_sim)
        rain_mask = rain_mask_regr & rain_mask_dist
        rain_i = var_names.index("R")
        for var_i, var_name in enumerate(var_names):
            print(var_name)
            if var_name == "R":
                continue
            try:
                npt.assert_almost_equal(
                    self.varwg_regr.data_trans[var_i],
                    self.varwg_dist.data_trans[var_i],
                )
            except AssertionError as exc:
                if self.verbose:
                    print(exc)
                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    ax.plot(self.varwg_regr.data_trans[var_i], label="regr")
                    ax.plot(self.varwg_dist.data_trans[var_i], label="dist")
                    ax.legend(loc="best")
                    ax.set_title(var_name)
                    plt.show()
                raise exc

        rain_trans_regr = self.varwg_regr.data_trans[rain_i]
        rain_trans_dist = self.varwg_dist.data_trans[rain_i]
        regr_dist, regr_sol = self.varwg_regr.dist_sol["R"]
        dist_dist, dist_sol = self.varwg_dist.dist_sol["R"]
        # doys = np.arange(365)
        # regr_params = regr_dist.all_parameters_dict(regr_sol, doys)
        # dist_params = dist_dist.all_parameters_dict(dist_sol, doys)

        # fig, axs = plt.subplots(nrows=len(regr_params), ncols=1)
        # for ax_i, (par_name, param) in enumerate(regr_params.items()):
        #     if par_name == "kernel_data":
        #         param = np.concatenate(param)
        #     axs[ax_i].plot(param)
        # # fig.set_prop_cycle(None)
        # for ax_i, (par_name, param) in enumerate(dist_params.items()):
        #     if par_name == "kernel_data":
        #         param = np.concatenate(param)
        #     axs[ax_i].plot(param)
        #     axs[ax_i].set_title(par_name)
        # plt.show()

        try:
            npt.assert_almost_equal(regr_sol.real, dist_sol.real, decimal=3)
        except AssertionError as exc:
            if self.verbose:
                print(exc)
                fig, axs = plt.subplots(
                    nrows=len(var_names), ncols=1, constrained_layout=True
                )
                for var_i, var_name in enumerate(var_names):
                    axs[var_i].plot(regr_sol[var_i].real, label="regr")
                    axs[var_i].plot(dist_sol[var_i].real, label="dist")
                    axs[var_i].set_title(var_name)
                ax.legend(loc="best")
                fig.suptitle("FFT parameters")
                plt.show()
                raise exc

        try:
            npt.assert_almost_equal(
                rain_trans_regr[rain_mask],
                rain_trans_dist[rain_mask],
                decimal=1,
            )
        except AssertionError as exc:
            if self.verbose:
                print(exc)
                fig, axs = plt.subplots(nrows=2, ncols=1)
                axs[0].plot(rain_trans_regr[rain_mask])
                axs[0].plot(rain_trans_dist[rain_mask])
                axs[1].scatter(
                    rain_trans_regr[rain_mask], rain_trans_dist[rain_mask]
                )
                plt.show()
                raise

    def test_plain(self):
        self.assertEqual(self.met_varwg.p, p)
        varwg.reseed(seed)
        sim_times, sim = self.met_varwg.simulate(T=T)
        sim = self.met_varwg.disaggregate(disagg_varnames)[1]
        try:
            assert np.all(np.isfinite(sim))
        except AssertionError as exc:
            if self.verbose:
                print(exc)
                import matplotlib.pyplot as plt

                # plt.plot(sim[0])
                self.met_varwg.plot_meteogram_hourly()
                # self.met_varwg.plot_meteogram_daily()
                plt.show()
                raise
        try:
            npt.assert_almost_equal(sim, self.sample_sim, decimal=2)
        except AssertionError:
            if self.verbose:
                import matplotlib.pyplot as plt

                fig, axs = plt.subplots(
                    nrows=self.met_varwg.K,
                    ncols=1,
                    sharex=True,
                    constrained_layout=True,
                )
                for k, ax in enumerate(axs):
                    ax.plot(self.sample_sim[k], label="sample")
                    ax.plot(sim[k], "--", label="sim")
                    ax.set_title(self.met_varwg.var_names[k])
                axs[0].legend(loc="best")
                plt.show()
                raise

    def test_seasonal(self):
        """Test only for exceptions, not for equality of results."""
        self.met_varwg.fit(p, seasonal=True)
        self.met_varwg.simulate(T)
        del self.met_varwg.Bs, self.met_varwg.sigma_us, self.met_varwg.seasonal

    # def test_extro(self):
    #     import matplotlib.pyplot as plt
    #     self.met_varwg.simulate()
    #     self.met_varwg.plot_autocorr()
    #     self.met_varwg.plot_VAR_par()
    #     print(np.round(self.met_varwg.AM, 2))

    #     self.met_varwg.fit(extro=True, p=p)
    #     self.met_varwg.plot_VAR_par()
    #     print(np.round(self.met_varwg.AM, 2))
    #     self.met_varwg.simulate()
    #     self.met_varwg.plot_autocorr()
    #     plt.show()

    # def test_multi_prim(self):
    #     # testing a change in humidity together with an increase in theta
    #     rh_signal, _ = self.met_varwg.random_dryness(
    #         duration_min=7, duration_max=14
    #     )
    #     varwg.reseed(seed)
    #     sim_times, sim, rphases = self.met_varwg.simulate(
    #         primary_var=("theta", "rh"),
    #         # theta_grad=(.43 * 3, None),
    #         climate_signal=(None, rh_signal),
    #         theta_incr=(4, None),
    #         return_rphases=True,
    #     )
    #     if self.verbose:
    #         fig, axs = self.met_varwg.plot_meteogram_daily()
    #     varwg.reseed(seed)
    #     self.met_varwg.simulate(
    #         primary_var=("theta", "rh"),
    #         theta_grad=(0.43 * 3, None),
    #         climate_signal=(None, rh_signal),
    #         theta_incr=(2, None),
    #         rphases=rphases,
    #     )

    # def test_rr_fact(self):
    #     r_fact = 1.5
    #     met_varwg = varwg.VarWG(("theta", "R", "Qsw"), data_dir=self.data_dir,
    #                    cache_dir=self.data_dir, met_file=met_file,
    #                    verbose=False)
    #     times, sim = met_varwg.simulate(r_fact=r_fact)
    #     r_index = met_varwg.var_names.index("R")
    #     npt.assert_almost_equal(np.nanmean(sim[r_index]),
    #                             np.nanmean(r_fact * met_varwg.data_raw[r_index] /
    #                                        met_varwg.sum_interval[r_index]))

    def test_resample(self):
        # use the constance data set here.  the self-generated data is
        # strange.... (hint: look what you have done there!)
        # import config_konstanz as conf
        # varwg.core.core.conf = varwg.conf = varwg.base.conf = conf
        met_varwg = varwg.VarWG(
            ("theta", "ILWR", "Qsw", "rh", "u", "v"),
            refit=self.refit,
            verbose=self.verbose,
        )
        met_varwg.fit(p=3)
        # shelve_filepath = os.path.join(varwg.conf.cache_dir,
        #                                varwg.resampler.shelve_filename)
        # if os.path.exists(shelve_filepath):
        #     os.remove(shelve_filepath)
        theta_incr = 4
        mean_arrival = 7
        disturbance_std = 5
        # kwds = dict(start_str="01.01.1994 00:00:00",
        #             stop_str="02.01.2015 00:00:00")
        kwds = dict()
        res_dict_nocy = my.ADict(recalibrate=True, n_candidates=20)
        res_dict_cy = res_dict_nocy + dict(cy=True)
        theta_mean = np.mean(met_varwg.data_raw[0] / met_varwg.sum_interval)
        for res_dict in (res_dict_nocy, res_dict_cy):
            kwds["res_kwds"] = res_dict
            # met_varwg.simulate(**kwds)
            simt, sim = met_varwg.simulate(theta_incr=theta_incr, **kwds)
            sim_diff = np.mean(sim[0]) - theta_mean
            npt.assert_almost_equal(sim_diff, theta_incr, decimal=0)
            simt, sim = met_varwg.simulate(
                mean_arrival=mean_arrival,
                disturbance_std=disturbance_std,
                **kwds,
            )
            sim_diff = np.mean(sim[0]) - theta_mean
            npt.assert_almost_equal(sim_diff, 0, decimal=0)
            # simt, sim = met_varwg.simulate(theta_incr=theta_incr,
            #                             mean_arrival=mean_arrival,
            #                             disturbance_std=disturbance_std,
            #                             **kwds)
            # sim_diff = np.mean(sim[0]) - theta_mean
            # npt.assert_almost_equal(sim_diff, theta_incr, decimal=0)

    def test_theta_incr(self):
        """Test theta increment with 20-year simulation."""
        # import config_konstanz
        # varwg.core.core.conf = varwg.conf = varwg.base.conf = config_konstanz
        # met_varwg = varwg.VarWG(("theta", "ILWR", "Qsw", "rh", "u", "v"),
        #                refit=True,
        #                verbose=self.verbose)
        self.met_varwg.fit(p=3, seasonal=True)
        theta_incr = 4
        mean_arrival = 7
        disturbance_std = 5
        T = int(365.25 * 20)
        theta_i = self.met_varwg.var_names.index("theta")
        data_mean = np.mean(self.met_varwg.data_raw[theta_i]) / 24.0
        simt, sim = self.met_varwg.simulate(T=T, theta_incr=theta_incr)
        npt.assert_almost_equal(
            sim[theta_i].mean() - data_mean, theta_incr, decimal=1
        )
        simt, sim = self.met_varwg.simulate(
            T=T,
            theta_incr=theta_incr,
            mean_arrival=mean_arrival,
            disturbance_std=disturbance_std,
        )
        npt.assert_almost_equal(
            sim[theta_i].mean() - data_mean, theta_incr, decimal=1
        )

    def test_theta_incr_nonnormal(self):
        """Test theta increment for non-normal variables."""
        varwg.reseed(seed)
        met_varwg = varwg.VarWG(
            ("R", "theta", "ILWR"),
            rain_method="simulation",
            refit=self.refit,
            verbose=self.verbose,
            dump_data=False,
        )
        met_varwg.fit(p=3, seasonal=True)
        theta_incr = 20
        simt, sim = met_varwg.simulate(
            theta_incr=theta_incr, primary_var="ILWR", T=100 * 365
        )
        prim_i = met_varwg.primary_var_ii[0]
        data_mean = np.mean(met_varwg.data_raw[prim_i]) / 24
        npt.assert_almost_equal(
            sim[prim_i].mean() - data_mean, theta_incr, decimal=1
        )

    # def test_rainmix(self):
    #     met_varwg = varwg.VarWG(
    #         # ("R", "theta", "ILWR"),
    #         var_names,
    #         # rain_method="regression",
    #         # rain_method="distance",
    #         rain_method="simulation",
    #         # refit="R",
    #         # refit=self.refit,
    #         refit=False,
    #         verbose=self.verbose,
    #         dump_data=False,
    #     )
    #     # fft_order = 2
    #     # doy_width = 15
    #     # met_varwg.fit(
    #     #     p=3, seasonal=True, fft_order=fft_order, doy_width=doy_width
    #     # )
    #     met_varwg.fit(**fit_kwds)

    #     # is it back-transforming right at all?
    #     simt, sim_back = met_varwg.simulate(
    #         residuals=met_varwg.residuals, phase_randomize=False
    #     )
    #     for var_i, (obs, sim) in enumerate(
    #         zip(met_varwg.data_raw / met_varwg.sum_interval, sim_back)
    #     ):
    #         try:
    #             npt.assert_almost_equal(sim, obs, decimal=1)
    #         except AssertionError as exc:
    #             if self.verbose:
    #                 var_name = met_varwg.var_names[var_i]
    #                 print(f"{var_name} does not backtransform to obs.")
    #                 print(exc)
    #                 fig, ax = plt.subplots(
    #                     nrows=1, ncols=1, subplot_kw=dict(aspect="equal")
    #                 )
    #                 ax.scatter(
    #                     obs, sim, s=5, facecolor=(0, 0, 0, 0), edgecolor="blue"
    #                 )
    #                 min_ = min(sim.min(), obs.min())
    #                 max_ = max(sim.max(), obs.max())
    #                 ax.plot([min_, max_], [min_, max_], "--", color="gray")
    #                 ax.axvline(
    #                     obs.mean(),
    #                     label=rf"$\overline{{obs}}$={obs.mean():.3f}",
    #                     color="black",
    #                 )
    #                 ax.axhline(
    #                     sim.mean(),
    #                     label=rf"$\overline{{sim}}$={sim.mean():.3f}",
    #                     color="blue",
    #                 )
    #                 if var_name == "R":
    #                     ax.axvline(
    #                         varwg.conf.threshold, linestyle="--", color="gray"
    #                     )
    #                     ax.axhline(
    #                         varwg.conf.threshold, linestyle="--", color="gray"
    #                     )
    #                 ax.set_xlabel("obs")
    #                 ax.set_ylabel("sim")
    #                 ax.set_title(var_name)
    #                 ax.legend(loc="best")
    #                 ax.grid(zorder=0)
    #                 fig, ax = plt.subplots(nrows=1, ncols=1)
    #                 ax.plot(obs, label="obs")
    #                 ax.plot(sim, label="sim")
    #                 if var_name == "R":
    #                     ax.axhline(
    #                         varwg.conf.threshold, linestyle="--", color="gray"
    #                     )
    #                 ax.legend(loc="best")
    #                 ax.grid(True, zorder=0)
    #                 ax.set_title(var_name)
    #             plt.show()
    #             raise

    #     varwg.reseed(seed)
    #     met_varwg.fit(p=3, seasonal=False)
    #     simt, sim = met_varwg.simulate(
    #         # T=100 * 365,
    #         phase_randomize=True,
    #         phase_randomize_vary_mean=False,
    #     )
    #     fig, axs = met_varwg.plot_seasonal_corrs(trans=True)
    #     fig.suptitle("nonseasonal")
    #     obs_trans_mean = np.mean(met_varwg.data_trans[0])
    #     sim_trans_mean = np.mean(met_varwg.sim[0])
    #     try:
    #         npt.assert_almost_equal(sim_trans_mean, obs_trans_mean, decimal=1)
    #     except AssertionError:
    #         if self.verbose:
    #             fig, axs = plt.subplots(
    #                 nrows=2,
    #                 ncols=1,
    #                 sharex=True,
    #                 sharey=True,
    #                 constrained_layout=True,
    #             )
    #             axs[0].plot(met_varwg.data_trans[0])
    #             axs[1].plot(met_varwg.sim[0], label="nonseasonal", color="blue")
    #             axs[1].axhline(
    #                 sim_trans_mean,
    #                 linestyle="--",
    #                 color="blue",
    #                 label="nonseasonal",
    #             )
    #             fq, axq = met_varwg.plot_qq(trans=True)
    #             met_varwg.fit(
    #                 p=3,
    #                 seasonal=True,
    #                 fft_order=fft_order,
    #                 doy_width=doy_width,
    #             )
    #             simt, sim2 = met_varwg.simulate(
    #                 phase_randomize=True,
    #                 phase_randomize_vary_mean=False,
    #             )
    #             figc, axsc = met_varwg.plot_seasonal_corrs(trans=True)
    #             figc.suptitle("seasonal")
    #             axs[1].plot(met_varwg.sim[0], label="seasonal", color="red")
    #             axs[1].axhline(
    #                 np.mean(met_varwg.sim[0]),
    #                 linestyle="--",
    #                 color="red",
    #                 label="seasonal",
    #             )
    #             axs[1].legend(loc="best")
    #             for ax in axs:
    #                 ax.grid(True, zorder=0)
    #                 ax.axhline(obs_trans_mean)
    #             met_varwg.plot_qq(trans=True, fig=fq, axs=axq, color="k")
    #         raise
    #     obs_mean = np.mean(met_varwg.data_raw[0] / met_varwg.sum_interval[0])
    #     sim_mean = np.mean(sim[0])
    #     # met_varwg.plot_meteogram_daily()
    #     # plt.show()
    #     npt.assert_almost_equal(sim_mean, obs_mean, decimal=1)

    #     # if self.verbose:
    #     #     figs, axss = met_varwg.plot_daily_fit("R")
    #     #     plt.show()

    #     # met_varwg.fit(3, seasonal=True)

    #     # import matplotlib.pyplot as plt
    #     # # import config_konstanz_disag as conf
    #     # import config_konstanz as conf
    #     # from varwg.core import plotting
    #     # varwg.conf = varwg.base.conf = varwg.plotting.conf = conf
    #     # varwg.reseed(seed)
    #     # met_varwg = varwg.VarWG(("R", "theta", "ILWR",
    #     #                 # "Qsw", "rh", "u", "v"
    #     #                 ),
    #     #                # non_rain=("theta", "Qsw", "rh"),
    #     #                # rain_method="distance",
    #     #                rain_method="regression",
    #     #                refit="R",
    #     #                # refit=True,
    #     #                verbose=self.verbose, dump_data=True)
    #     # met_varwg.fit(3)
    #     # fig, ax = plt.subplots(nrows=1, ncols=1)
    #     # r_dist, sol = met_varwg.dist_sol["R"]
    #     # x_evals = [suppl["x_eval"] for suppl in r_dist.supplements]
    #     # ax.violinplot(x_evals)
    #     # ax.plot(met_varwg.data_doys, met_varwg.data_raw[0])
    #     # # plt.show()
    #     # simt, sim = met_varwg.simulate()
    #     # # simt_dis, sim_dis = met_varwg.disaggregate()
    #     # met_varwg.plot_meteogram_trans()
    #     # met_varwg.plot_meteogram_daily()
    #     # # met_varwg.plot_daily_fit("R")
    #     # fig, axs = met_varwg.plot_exceedance_daily()
    #     # fig.suptitle("regression")
    #     # fig, axs = met_varwg.plot_qq()
    #     # fig.suptitle("regression")
    #     # varwg.reseed(seed)
    #     # met_varwg = varwg.VarWG(("R", "theta", "ILWR",
    #     #                 # "Qsw", "rh", "u", "v"
    #     #                 ),
    #     #                # non_rain=("theta", "Qsw", "rh"),
    #     #                rain_method="distance",
    #     #                # rain_method="regression",
    #     #                refit="R",
    #     #                # refit=True,
    #     #                verbose=self.verbose, dump_data=True)
    #     # met_varwg.fit(3)
    #     # simt, sim = met_varwg.simulate()
    #     # # simt_dis, sim_dis = met_varwg.disaggregate()
    #     # # met_varwg.plot_meteogram_daily()
    #     # fig, axs = met_varwg.plot_exceedance_daily()
    #     # fig.suptitle("distance")
    #     # fig, axs = met_varwg.plot_qq()
    #     # fig.suptitle("distance")
    #     # plt.show()

    # def test_rm(self):
    #     import config_konstanz_disag
    #     varwg.conf = varwg.base.conf = config_konstanz_disag
    #     met_varwg = varwg.VarWG(("theta", "ILWR", "Qsw", "rh", "u", "v"),
    #                    verbose=True)
    #     met_varwg.simulate()
    #     met_varwg.disaggregate_rm()

    def test_to_df(self):
        self.met_varwg.simulate()
        self.met_varwg.disaggregate(disagg_varnames)
        self.met_varwg.to_df("hourly input")
        self.met_varwg.to_df("daily input")
        self.met_varwg.to_df("daily output")
        self.met_varwg.to_df("hourly output")

    def test_phase_randomization(self):
        for rain_method in ("regression", "distance"):
            met_varwg = varwg.VarWG(
                ("R", "theta", "ILWR"),
                # rain_method="distance",
                rain_method="regression",
                # refit=self.refit,
                verbose=self.verbose,
                dump_data=False,
            )
            met_varwg.fit(p=3)
            simt, sim = met_varwg.simulate(
                T=1001, phase_randomize=True, phase_randomize_vary_mean=True
            )
            try:
                self.assertEqual(sim.shape[1], 1001)
            except AssertionError:
                if self.verbose:
                    met_varwg.plot_meteogram_daily()
                    plt.show()
                raise
            simt, sim = met_varwg.simulate(
                T=1000, phase_randomize=True, phase_randomize_vary_mean=True
            )
            try:
                self.assertEqual(sim.shape[1], 1000)
            except AssertionError:
                if self.verbose:
                    met_varwg.plot_meteogram_daily()
                    plt.show()
                raise
            simt, sim = met_varwg.simulate(
                T=met_varwg.data_trans.shape[1],
                phase_randomize=True,
                phase_randomize_vary_mean=True,
            )
            try:
                self.assertEqual(sim.shape[1], met_varwg.data_trans.shape[1])
            except AssertionError:
                if self.verbose:
                    met_varwg.plot_meteogram_daily()
                    plt.show()
                raise
            simt, sim = met_varwg.simulate(
                T=met_varwg.data_trans.shape[1] + 1,
                phase_randomize=True,
                phase_randomize_vary_mean=True,
            )
            try:
                self.assertEqual(
                    sim.shape[1], met_varwg.data_trans.shape[1] + 1
                )
            except AssertionError:
                if self.verbose:
                    met_varwg.plot_meteogram_daily()
                    plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    import warnings

    warnings.simplefilter("error", RuntimeWarning)
    # if os.path.exists(met_file):
    #     npt.run_module_suite()
