from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()
from builtins import range
from past.builtins import basestring
from past.utils import old_div
import collections
from future.moves.dbm import dumb
import os
from tempfile import mkstemp
import warnings
import numpy as np
from scipy import stats
from matplotlib import gridspec, ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
from vg.meteo.windrose import windrose, seasonal_windroses
from vg.meteo import avrwind, meteox2y
from vg.core import vg_base
import vg.time_series_analysis.seasonal_distributions as sd
import vg.time_series_analysis.time_series as ts
from vg.time_series_analysis import rain_stats
from vg import times
from vg import helpers as my

try:
    from vg import config as conf
except ImportError:
    from vg import config_template as conf
    conf_filepath = conf.__file__
    if conf_filepath.endswith(".pyc"):
        conf_filepath = conf_filepath[:-1]
    # warnings.warn('Could not import "config.py".\n' +
    #               'Edit "%s" and rename it to "config.py"' % conf_filepath)


def append_fa(fig_ax, figs=None, axs=None):
    if figs is None:
        figs = []
    if axs is None:
        axs = []
    assert len(figs) == len(axs)
    fig, ax = fig_ax
    return figs + [fig], axs + [ax]


def var_names_greek(var_names):
    return [conf.ygreek[var_name] for var_name in var_names]


class VGPlotting(vg_base.VGBase):

    """To be used indirectly (by the VG class) or directly to plot
    meteorological data (met_files, pandas dataframe).

    Caching features will be disabled to not overwrite anything more useful.
    """

    def __init__(self, var_names, met_file=None, sum_interval=24, max_nans=12,
                 plot=False, separator="\t", refit=None,
                 detrend_vars=None, verbose=False, data_dir=None,
                 cache_dir=None, seasonal_fitting=False,
                 dump_data=False, non_rain=None, rain_method=None,
                 **met_kwds):
        # signal for VGBase to not do the seasonal fitting, which takes time
        # and the reason to initiate VGPlotting might be that you just plot
        # something quickly
        self.seasonal_fitting = seasonal_fitting
        super(VGPlotting, self).__init__(var_names, met_file=met_file,
                                         sum_interval=sum_interval,
                                         plot=plot,
                                         separator=separator,
                                         refit=refit,
                                         detrend_vars=detrend_vars,
                                         verbose=verbose,
                                         data_dir=data_dir,
                                         cache_dir=cache_dir,
                                         dump_data=dump_data,
                                         non_rain=non_rain,
                                         rain_method=rain_method,
                                         max_nans=max_nans,
                                         **met_kwds)

    def plot_all(self, *args, **kwds):
        """Executes every method of VGPlotting that starts with "plot_".
        There are quite a number of those. You have been warned.
        """
        methods = {attr: getattr(self, attr)
                   for attr in dir(VGPlotting)
                   if (attr.startswith("plot_")
                       and attr != "plot_all")}
        return_values = []
        for meth_name, meth in list(methods.items()):
            if self.verbose:
                print("Calling %s" % meth_name)
            try:
                fig_ax = meth(*args, **kwds)
                return_values += [fig_ax]
            except Exception as exc:
                warnings.warn("Exception while calling %s:\n%s" %
                              (meth, exc))
                raise
        return return_values

    def fit_seasonal(self, refit=None, values=None, doys=None):
        """This is here, so that when initialized directly, seasonal
        distributions are fitted, but the existing fit to the data
        defined in the config file is not overwritten.
        """
        f_handle, self.seasonal_cache_file = mkstemp(suffix=".she")
        # need to establish the fact that the temp file is a data base
        db = dumb.open(self.seasonal_cache_file, "n")
        db.close()
        os.close(f_handle)
        self.data_trans, self.dist_sol = \
            super(VGPlotting, self)._fit_seasonal(refit, values, doys)

    def __getattribute__(self, name):
        if name == "dist_sol" and not self.seasonal_fitting:
            # ok, we need the seasonal fitting, so provide it
            self.seasonal_fitting = True
            self.fit_seasonal(refit="all")
            return self.dist_sol
        return vg_base.VGBase.__getattribute__(self, name)

    @property
    def ylabels(self):
        return [conf.units[var] if var in conf.units else "[?]"
                for var in self.var_names]

    def plot_exceedance_daily(self, thresh=None, fig=None, axs=None,
                            *args, **kwds):
        if "R" not in self.var_names:
            warnings.warn("No R in var_names, no plot_exceedance_daily")
            return
        if thresh is None:
            thresh = conf.threshold
        rain_i = self.var_names.index("R")
        sim = self.sim_sea[rain_i]
        obs = self.data_raw[rain_i] / self.sum_interval[rain_i]
        fig, axs = rain_stats.plot_exceedance(obs, sim, kind="all",
                                              thresh=thresh, fig=fig,
                                              axs=axs, *args, **kwds)
        title_str = "Precipitation exceedance"
        fig.suptitle(title_str)
        try:
            fig.canvas.set_window_title(f"{title_str} ({fig.canvas.manager.num})")
        except AttributeError:
            pass
        return fig, axs

    def plot_exeedance_hourly(self, thresh=None, fig=None, axs=None,
                            *args, **kwds):
        if "R" not in self.var_names:
            warnings.warn("No R in var_names, no plot_exceedance_hourly")
            return
        if self.sim_sea is None:
            warnings.warn("Call simulate first.")
            return
        if self.sim_sea_dis is None:
            warnings.warn("No hourly simulation available")
            return
        if thresh is None:
            thresh = conf.threshold
        rain_i = self.var_names.index("R")
        sim = self.sim_sea_dis[rain_i]
        obs = self.met["R"]
        return rain_stats.plot_exceedance(obs, sim, kind="all",
                                     thresh=thresh, fig=None, axs=None,
                                     *args, **kwds)

    def plot_hyd_year_sums(self, fig=None, ax=None, alpha=0.5, *args, **kwds):
        if "R" not in self.var_names:
            warnings.warn("No R in var_names, no plot_hyd_year_sums")
            return
        import xarray as xr
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            first_fig = True
        else:
            first_fig = False
        obs_xr = (xr
                  .DataArray(self.to_df("daily input"))
                  .rename(dict(dim_1="variable"))
                  .sel(variable="R"))
        obs = np.sort(rain_stats.hyd_year_sums(obs_xr, full_years=True))
        ax.step(obs, my.rel_ranks(len(obs)), where="mid", label="obs")

        if self.sim_sea is not None:
            sim_xr = (xr
                      .DataArray(self.to_df("daily output"))
                      .rename(dict(dim_1="variable"))
                      .sel(variable="R"))
            sim = np.sort(rain_stats.hyd_year_sums(sim_xr, full_years=True))
            ax.step(sim, my.rel_ranks(len(sim)), where="mid",
                    color="k", alpha=alpha, label="sim")
            if first_fig:
                ax.legend(loc="best")

        if first_fig:
            ax.grid(True)
            ax.set_ylabel("cdf")
            ax.set_xlabel(f"R {conf.units['R']}")
            title_str = "Hydrological year sums (cdf)"
            ax.set_title(title_str)
            try:
                fig.canvas.set_window_title(f"{title_str} ({fig.canvas.manager.num})")
            except AttributeError:
                pass
        return fig, ax

    def plot_psd(self, var_names=None, hourly=False, *args, **kwds):
        """Plots power spectral density using matplotlib.pyplot.psd.

        Parameter
        ---------
        var_names : sequence of str, optional
            Which variables to plot. None means plot all.
        hourly : boolean, optional
            Plot hourly and disaggregated values.
        *args, **kwds :
            Passed on to matplotlib.pyplot.psd
        """
        if var_names is None:
            var_names = self.var_names

        fig, axs = plt.subplots(len(var_names))
        if hourly:
            data_obs = [self.met[var_name] for var_name in var_names]
            data_sim = [self.sim_sea_dis[self.var_names.index(var_name)]
                        for var_name in var_names]
            # time discretization
            dt = (self.dis_times[1] - self.dis_times[0]).total_seconds()
            suptitle = "Power spectral density - hourly"
            name = "psd_hourly"
        else:
            data_obs = [self.data_raw[self.var_names.index(var_name)]
                        for var_name in var_names]
            data_sim = [self.sim_sea[self.var_names.index(var_name)]
                        for var_name in var_names]
            dt = (self.sim_times[1] - self.sim_times[0]).total_seconds()
            suptitle = "Power spectral density - daily"
            name = "psd_daily"

        for var_name in var_names:
            var_i = self.var_names.index(var_name)
            axs[var_i].psd(data_obs[var_i], Fs=old_div(1, dt),
                           scale_by_freq=True, label="observed",
                           *args, **kwds)
            axs[var_i].psd(data_sim[var_i], Fs=old_div(1, dt),
                           scale_by_freq=True, label="simulated",
                           *args, **kwds)
            axs[var_i].set_xscale('log')
            axs[var_i].set_ylabel("%s [dB/Hz]" % var_name)
        axs[0].legend(loc="best")
        fig.suptitle(suptitle)
        fig.name = name

        return fig, axs

    def plot_doy_scatter(self, var_names=None, opacity=.4, hourly=False):
        """Plot variables over doys. Measurement and backtransformed simulated
        data will be plotted, if it is available.

        Parameters
        ----------
        var_names : sequence of str, optional
            Which variables to plot. None means plot all.
        opacity : float in the range (0,1], optional
            Opacity of the scatter points.

        .. plot::

            met_vg = VG(("theta",))
            sim_times, sim = met_vg.simulate()
            met_vg.plot_doy_scatter()

        """
        if var_names is None:
            var_names = self.var_names
        elif isinstance(var_names, basestring):
            var_names = var_names,
        fig, axs = plt.subplots(nrows=len(var_names))
        if len(var_names) == 1:
            axs = axs,
        for ax, var_name in zip(axs, var_names):
            var_ii = self.var_names.index(var_name)
            if hourly:
                ax.scatter(times.datetime2doy(self.times_orig),
                           self.met[var_name],
                           marker="o", edgecolors=(0, 0, 1, opacity),
                           facecolors=(0, 0, 0, 0), label="measured")
            else:
                ax.scatter(self.data_doys,
                           self.data_raw[var_ii] / self.sum_interval[var_ii],
                           marker="o", edgecolors=(0, 0, 1, opacity),
                           facecolors=(0, 0, 0, 0), label="measured")

            if self.sim_sea is not None:
                if hourly:
                    ax.scatter(self.dis_doys, self.sim_sea_dis[var_ii],
                               marker="o", edgecolors=(1, 0, 0, opacity),
                               facecolors=(0, 0, 0, 0),
                               label="simulated backtransformed")
                else:
                    ax.scatter(self.sim_doys, self.sim_sea[var_ii],
                               marker="o", edgecolors=(1, 0, 0, opacity),
                               facecolors=(0, 0, 0, 0),
                               label="simulated backtransformed")
            ax.set_xlabel("doy")
            ax.set_ylabel("%s %s" %
                          (var_names_greek(self.var_names)[var_ii],
                           conf.units[var_name]))
            ax.legend(loc="best")
            ax.grid(True)
            ax.set_xlim(0, 366)
            ax.set_title(conf.long_var_names[var_name])
        title_str = "Seasonal scatter"
        fig.suptitle(title_str)
        try:
            fig.canvas.set_window_title(f"{title_str} ({fig.canvas.manager.num})")
        except AttributeError:
            pass
        return fig, ax

    def _meteogram(self, time, data, suptitle, var_names, fig=None, axs=None,
                 plot_dewpoint=True, p_kwds=None, h_kwds=None, **f_kwds):
        if fig is None:
            fig = plt.figure(constrained_layout=True, **f_kwds)
        if var_names is None:
            var_names = self.var_names
        if isinstance(var_names, basestring):
            var_names = var_names,
        if p_kwds is None:
            p_kwds = {}
        if h_kwds is None:
            h_kwds = {}
        K = len(var_names)
        if axs is None:
            gs = gridspec.GridSpec(K, 2, figure=fig, width_ratios=[8, 1])
            axs = np.empty((K, 2), dtype=object)
            for i in range(K):
                if i == 0:
                    axs[0, 0] = plt.subplot(gs[0, 0])
                else:
                    axs[i, 0] = plt.subplot(gs[i, 0], sharex=axs[i - 1, 0])
                axs[i, 1] = plt.subplot(gs[i, 1], sharey=axs[i, 0])

        for var_name in var_names:
            var_i = var_names.index(var_name)
            plt.xticks(rotation=70)
            axs[var_i, 0].plot(time, data[var_i], label=var_name,
                               **p_kwds)
            if var_name == "theta" and ("rh" in var_names
                                      or "abs_hum" in var_names):
                theta = data[var_i]
                if "rh" in var_names:
                    rh = data[var_names.index("rh")]
                else:
                    abs_hum = data[var_names.index("abs_hum")]
                    rh = meteox2y.abs_hum2rel(abs_hum, theta)
                dewpoint = meteox2y.dewpoint(theta, rh=rh)
                axs[var_i, 0].plot(time, dewpoint, "--",
                                   label="dewpoint", color="k", alpha=0.5,
                                   **p_kwds)
            if var_name == "abs_hum" and "theta" in var_names:
                theta = data[var_names.index("theta")]
                sat_vap_p = meteox2y.sat_vap_p(theta)
                axs[var_i, 0].plot(time, sat_vap_p, "--",
                                   label="sat_vap_p", color="k",
                                   alpha=0.5, **p_kwds)
            axs[var_i, 0].grid(True)
            axs[var_i, 0].set_ylabel("%s %s" % (conf.ygreek[var_name],
                                                conf.units[var_name]))
            finite_mask = np.isfinite(data[var_i])
            try:
                axs[var_i, 1].hist(data[var_i, finite_mask], 40,
                                   density=True, histtype="step",
                                   orientation="horizontal",
                                   **h_kwds)
            except TypeError:
                pass
            # if var_i < (K - 1):
            #     axs[var_i, 0].set_xticklabels("")
            axs[var_i, 1].set_xticklabels("")
        plt.suptitle(suptitle)
        return fig, axs

    def _add_daily_bounds(self, axs, var_names, doys, times):
        keys = "u", "l", "uc", "lc"
        for var_i, ax in enumerate(axs[:, 0]):
            var_name = var_names[var_i]
            if var_name not in conf.par_known:
                continue
            pars = conf.par_known[var_name]
            if pars is None:
                continue
            for key in keys:
                if key not in pars:
                    continue
                par_values = (pars[key](doys)
                              / self.sum_interval_dict[var_name])
                ax.plot(times, par_values, label=key)
            ax.legend(loc="best")
        return axs

    def plot_meteogram_daily(self, var_names=None, figs=None, axss=None, figsize=None,
                           plot_sim_sea=True, obs=None, sim=None,
                           plot_daily_bounds=True, p_kwds=None, h_kwds=None,
                           **f_kwds):
        if var_names is None:
            var_names = self.var_names
        if figs is None:
            figs = 2 * [None]
        elif isinstance(figs, mpl.figure.Figure):
            figs = figs,
        if axss is None:
            axss = 2 * [None]
        if figsize is not None:
            f_kwds["figsize"] = figsize
        # elif isinstance(axss, np.ndarray):
        #     axss = axss,
        if self.ex_in is None and obs is None:
            obs = self.data_raw / self.sum_interval
        elif obs is None:
            obs = np.vstack((self.data_raw / self.sum_interval, self.ex_in))
            if "R" in var_names:
                rain_i = var_names.index("R")
                obs[rain_i] *= self.sum_interval[rain_i]
            var_names = list(var_names) + ["external"]

        fig, axs = self._meteogram(self.times, obs,
                                   "Measured daily",
                                   var_names, fig=figs[0],
                                   axs=axss[0], p_kwds=p_kwds,
                                   h_kwds=h_kwds, **f_kwds)
        fig.name = "meteogram_measured_daily"
        if plot_daily_bounds:
            axs = self._add_daily_bounds(axs, var_names,
                                         self.data_doys, self.times)

        if plot_sim_sea and self.sim_sea is not None:
            if self.ex_out is None and sim is None:
                sim = self.sim_sea
            elif sim is None:
                sim = np.vstack((self.sim_sea, self.ex_out))
                var_names = list(var_names) + ["external"]
            fig_, axs_ = self._meteogram(self.sim_times, sim,
                                         "Simulated daily", var_names,
                                         fig=figs[1], axs=axss[1],
                                         p_kwds=p_kwds, h_kwds=h_kwds,
                                         **f_kwds)
            fig_.name = "meteogram_sim_daily"
            if plot_daily_bounds:
                axs_ = self._add_daily_bounds(axs_, var_names,
                                              self.sim_doys,
                                              self.sim_times)
            fig, axs = np.array([fig, fig_]), np.array([axs, axs_])
        return fig, axs

    def plot_meteogram_hourly(self, var_names=None, figs=None, axss=None,
                            plot_sim_sea=True, obs=None, sim=None,
                            combine=False, **f_kwds):
        """All variables over time in subplots.
        (Not as nice as the meteogram from vg.meteo)
        """
        if figs is None:
            figs = 2 * [None]
        if axss is None:
            axss = 2 * [None]
        if obs is None:
            if var_names is None:
                var_names = self.var_names
            obs = vg_base.met_as_array(self.met, var_names=var_names)
        fig, axs = self._meteogram(self.times_orig, obs,
                                   "Measured hourly", var_names,
                                   fig=figs[0], axs=axss[0])
        fig.name = "meteogram_measured_houry"

        if plot_sim_sea and self.sim_sea_dis is not None:
            if self.ex_out is None and sim is None:
                sim = self.sim_sea_dis
                var_names = self.var_names
            elif sim is None:
                sim = np.vstack((self.sim_sea_dis, self.ex_out))
                var_names = list(self.var_names) + ["external"]
            if combine:
                fig, axs = self._meteogram(self.dis_times,
                                           sim,
                                           "Simulated daily",
                                           var_names, fig=fig,
                                           axs=axs)
            else:
                fig_, axs_ = self._meteogram(self.dis_times,
                                             sim,
                                             "Simulated daily",
                                             var_names,
                                             fig=figs[1],
                                             axs=axss[1])
                fig_.name = "meteogram_sim_daily"
                fig, axs = np.array([fig, fig_]), np.array([axs, axs_])

        # if self.sim_sea_dis is not None:
        #     fig_, axs_ = self._meteogram(self.dis_times, self.sim_sea_dis,
        #                                   "Simulated hourly", var_names,
        #                                   fig=figs[1], axs=axss[1],
        #                                   station_name=self.station_name)
        #     fig_.name = "meteogram_sim_hourly"
        #     fig, axs = np.array([fig, fig_]), np.array([axs, axs_])
        return fig, axs

    def plot_meteogram_trans(self, var_names=None, figs=None, axss=None,
                           **f_kwds):
        if figs is None:
            figs = 2 * [None]
        elif isinstance(figs, mpl.figure.Figure):
            figs = figs,
        if axss is None:
            axss = 2 * [None]
        elif isinstance(axss, np.ndarray):
            axss = axss,
        fig, axs = self._meteogram(self.times, self.data_trans,
                                   "Observed transformed", var_names,
                                   fig=figs[0], axs=axss[0])
        fig.name = "meteogram_measured_daily_trans"
        if self.sim is not None:
            fig_, axs_ = self._meteogram(self.sim_times, self.sim,
                                         "Simulated std-normal",
                                         var_names, fig=figs[1],
                                         axs=axss[1])
            fig_.name = "meteogram_sim_trans"
            fig, axs = np.array([fig, fig_]), np.array([axs, axs_])
        return fig, axs

    def plot_doy_scatter_residuals(self, var_names=None, opacity=.4):
        if var_names is None:
            var_names = self.var_names
        # when given a string, this does not separate the letters, as tuple()
        # or list does
        var_names = np.atleast_1d(var_names)
        figs = []
        axs = []
        for var_name in var_names:
            var_ii = self.var_names.index(var_name)
            fig, ax = plt.subplots(nrows=1, ncols=1)
            figs.append(fig)
            axs.append(ax)
            ax.scatter(self.data_doys,
                       self.data_trans[var_ii], marker="o",
                       edgecolors=(0, 0, 1, opacity),
                       facecolors=(0, 0, 0, 0), label="transformed")

            ax.scatter(self.data_doys, self.residuals[var_ii], marker="o",
                       edgecolors=(1, 0, 0, opacity),
                       facecolors=(0, 0, 0, 0),
                       label="residuals")
            ax.set_xlabel("doy")
            ax.set_ylabel("%s %s" % (var_names_greek(self.var_names)[var_ii],
                                     conf.units[var_name]))
            ax.legend()
            ax.grid()
            title_str = ("Seasonal scatter of residuals %s" %
                         conf.long_var_names[var_name])
            ax.set_title(title_str)
            try:
                fig.canvas.set_window_title(f"{title_str} ({fig.canvas.manager.num})")
            except AttributeError:
                pass
        return figs, axs

    def plot_spaces(self, var_names=None, opacity=.4):
        if var_names is None:
            var_names = self.var_names

        figs = []
        axs = []
        for var_name1 in var_names:
            var_ii = self.var_names.index(var_name1)
            for var_jj in range(var_ii + 1, len(var_names)):
                var_name2 = var_names[var_jj]
                fig, ax = plt.subplots(nrows=1, ncols=1,
                                       subplot_kw=dict(aspect="equal"))
                figs.append(fig)
                axs.append(ax)
                ax.scatter(my.rel_ranks(self.data_trans[var_ii]),
                           my.rel_ranks(self.data_trans[var_jj]),
                           marker="o", edgecolors=(0, 0, 1, opacity),
                           facecolors=(0, 0, 0, 0), label="transformed data")
                ax.scatter(my.rel_ranks(self.sim[var_ii]),
                           my.rel_ranks(self.sim[var_jj]), marker="o",
                           edgecolors=(1, 0, 0, opacity),
                           facecolors=(0, 0, 0, 0), label="simulated")
                ax.legend()
                ax.set_xlabel(var_name1)
                ax.set_ylabel(var_name2)
                ax.grid()
                title_str = ("Spaces %s over %s" %
                             (conf.long_var_names[var_name2],
                              conf.long_var_names[var_name1]))
                ax.set_title(title_str)
                try:
                    fig.canvas.set_window_title(f"{title_str} ({fig.canvas.manager.num})")
                except AttributeError:
                    pass
        return figs, axs

    def plot_diff_spaces(self, var_names=None, diff=1, opacity=.4):
        if var_names is None:
            var_names = self.var_names

        figs = []
        axs = []
        for var_name in var_names:
            var_ii = self.var_names.index(var_name)
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   subplot_kw=dict(aspect="equal"))
            figs.append(fig)
            axs.append(ax)
            ax.scatter(my.rel_ranks(self.data_trans[var_ii, diff:]),
                       my.rel_ranks(self.data_trans[var_ii, :-diff]),
                       marker="o", edgecolors=(0, 0, 1, opacity),
                       facecolors=(0, 0, 0, 0),
                       label="transformed data")
            ax.scatter(my.rel_ranks(self.sim[var_ii, diff:]),
                       my.rel_ranks(self.sim[var_ii, :-diff]),
                       marker="o", edgecolors=(1, 0, 0, opacity),
                       facecolors=(0, 0, 0, 0), label="simulated")
            ax.legend()
            ax.set_xlabel("t")
            ax.set_ylabel("t - %d" % diff)
            ax.grid(True)
            title_str = ("Spaces %d-lagged %s" %
                         (diff, conf.long_var_names[var_name]))
            ax.set_title(title_str)
            try:
                fig.canvas.set_window_title(f"{title_str} ({fig.canvas.manager.num})")
            except AttributeError:
                pass
        return figs, axs

    def plot_hourly_fit(self, var_names=None, *args, **kwds):
        """Plots hourly scatter with fitted densities and a histogram of the
        transformed quantiles.

        Parameters
        ----------
        var_names : sequence of str, optional
            Which variables to plot. None means plot all.
        plot_quantiles : boolean, optional
            Plot histograms of the quantiles?
        plot_fourier : boolean, optional
            Plot the fourier fit to the seasonally changing distribution
            parameters.
        plot_monthly : boolean, optional
            Plot histograms showing how good the fit is on monthly separated
            data.
        opacity : float, optional
            Opacity of scatter.
        n_bins : int or sequence, optional
            Passed on to np.histogram when plotting histograms of quantiles.
        kde : boolean, optional
            Plot a kernel density estimation on the histograms of quantiles.
        s_kwds : None or dict, optional
            Keyword arguments passed on to plt.scatter.
        """
        return self._plot_seasonal_fit(var_names=None, hourly=True, *args,
                                       **kwds)

    def plot_daily_fit(self, var_names=None, *args, **kwds):
        """Plots daily scatter with fitted densities and a histogram of the
        transformed quantiles.

        Parameters
        ----------
        var_names : sequence of str, optional
            Which variables to plot. None means plot all.
        plot_quantiles : boolean, optional
            Plot histograms of the quantiles?
        plot_fourier : boolean, optional
            Plot the fourier fit to the seasonally changing distribution
            parameters.
        plot_monthly : boolean, optional
            Plot histograms showing how good the fit is on monthly separated
            data.
        opacity : float, optional
            Opacity of scatter.
        n_bins : int or sequence, optional
            Passed on to np.histogram when plotting histograms of quantiles.
        kde : boolean, optional
            Plot a kernel density estimation on the histograms of quantiles.
        s_kwds : None or dict, optional
            Keyword arguments passed on to plt.scatter.
        """
        return self._plot_seasonal_fit(var_names=var_names, hourly=False,
                                  *args, **kwds)

    def _plot_seasonal_fit(self, var_names=None, plot_quantiles=True,
                         plot_fourier=True, plot_monthly=False,
                         plot_seasonality=False, hourly=False,
                         opacity=.25, n_bins=15, kde=True, figs=None,
                         axss=None, s_kwds=None, *args, **kwds):
        """Plots seasonal scatter with fitted densities and a histogram of the
        transformed quantiles.

        Parameters
        ----------
        var_names : sequence of str, optional
            Which variables to plot. None means plot all.
        plot_quantiles : boolean, optional
            Plot histograms of the quantiles?
        plot_fourier : boolean, optional
            Plot the fourier fit to the seasonally changing distribution
            parameters.
        plot_monthly : boolean, optional
            Plot histograms showing how good the fit is on monthly separated
            data.
        opacity : float, optional
            Opacity of scatter.
        n_bins : int or sequence, optional
            Passed on to np.histogram when plotting histograms of quantiles.
        kde : boolean, optional
            Plot a kernel density estimation on the histograms of quantiles.
        s_kwds : None or dict, optional
            Keyword arguments passed on to plt.scatter.
        """
        if var_names is None:
            var_names = self.var_names
        elif isinstance(var_names, basestring):
            var_names = var_names,

        if figs is None:
            figs = []
        if axss is None:
            axss = []
        for var_name in var_names:
            var_ii = self.var_names.index(var_name)
            if hourly:
                try:
                    dist, solution = self.dist_sol_hourly(var_name)
                except KeyError:
                    warnings.warn("No hourly fit for %s available" % var_name)
                    continue
                n_sumup = 1
                values = self.met[var_name]
                doys = self.data_doys_raw
                discr_str = "hourly"
            else:
                dist, solution = self.dist_sol[var_name]
                n_sumup = self.sum_interval.ravel()[var_ii]
                values = self.data_raw[var_ii]
                doys = self.data_doys
                discr_str = "daily"

            if var_name in conf.long_var_names:
                long_var_name = conf.long_var_names[var_name]
            else:
                long_var_name = var_name

            # scatter pdf
            fig, ax = dist.scatter_pdf(solution, title=long_var_name,
                                       n_sumup=n_sumup, opacity=opacity,
                                       s_kwds=s_kwds, *args, **kwds)
            if self.sim_sea is not None:
                # data_sim = (self.sim_sea[var_ii] *
                #             self.sum_interval_dict[var_name])
                data_sim = self.sim_sea[var_ii]
                ax.scatter(self.sim_doys, data_sim,
                           marker="x", facecolor=(0, 0, 0, opacity))
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(min(ymin, data_sim.min()),
                            max(ymax, data_sim.max()))
            fig.name = "scatter_pdf_%s_%s" % (var_name, discr_str)
            ax.set_ylabel(conf.units[var_name]
                          if var_name in conf.units else "[?]")
            title_str = ("Scatter pdf %s (%d) %s"
                         % (var_name,
                            fig.canvas.manager.num,
                            discr_str))
            ax.set_title(title_str)
            try:
                fig.canvas.set_window_title(title_str)
            except AttributeError:
                pass
            figs += [fig]
            axss += [ax]

            if plot_quantiles:
                fig = plt.figure(*args, **kwds)
                var_ii = self.var_names.index(var_name)
                quantiles = dist.cdf(solution, x=values, doys=doys)
                my.hist(quantiles[np.isfinite(quantiles)],
                        n_bins, fig=fig, kde=kde)
                title_str = ("Quantiles %s (%d) %s"
                             % (var_name,
                                fig.canvas.manager.num,
                                discr_str))
                plt.title(title_str)
                try:
                    fig.canvas.set_window_title(title_str)
                except AttributeError:
                    pass
                fig.name = ("scatter_pdf_quantiles_%s_%s" %
                            (var_name, discr_str))
                figs += [fig]

            suptitle = "%s %s" % (var_name, discr_str)
            if plot_fourier:
                if isinstance(dist, sd.SeasonalDist):
                    fig, _ = dist.plot_fourier_fit()
                    fig.suptitle(suptitle)
                    try:
                        fig.canvas.set_window_title(suptitle)
                    except AttributeError:
                        pass
                    fig.name = "scatter_pdf_fft_%s_%s" % (var_name, discr_str)
                    figs += [fig]
            if plot_monthly:
                if isinstance(dist, sd.SlidingDist):
                    fig, axs = dist.plot_monthly_fit()
                    try:
                        for f in fig:
                            f.suptitle(suptitle)
                            try:
                                fig.canvas.set_window_title(suptitle)
                            except AttributeError:
                                pass
                    except TypeError:
                        fig.suptitle(suptitle)
                        try:
                            fig.canvas.set_window_title(suptitle)
                        except AttributeError:
                            pass
                    if isinstance(fig, mpl.figure.Figure):
                        fig = [fig]
                    if isinstance(axs, np.ndarray):
                        axs = axs.tolist()
                    figs += fig
                    axss += axs
            if plot_seasonality:
                if isinstance(dist, sd.SlidingDist):
                    fig, axs = dist.plot_seasonality_fit()
                    try:
                        for f in fig:
                            f.suptitle(suptitle)
                            try:
                                fig.canvas.set_window_title(suptitle)
                            except AttributeError:
                                pass
                            
                    except TypeError:
                        fig.suptitle(suptitle)
                        try:
                            fig.canvas.set_window_title(suptitle)
                        except AttributeError:
                            pass
                    if isinstance(fig, mpl.figure.Figure):
                        fig = [fig]
                    if isinstance(axs, np.ndarray):
                        axs = axs.tolist()
                    figs += fig
                    axss += axs
        return figs, axss

    def plot_qq(self, var_names=None, figsize=None, lines=True, trans=False,
              color="gray", fig=None, axs=None, *args, **kwds):
        if var_names is None:
            var_names = self.var_names
        if isinstance(var_names, basestring):
            var_names = var_names,
        n_axes = len(var_names)
        n_cols = int(np.ceil(n_axes ** .5))
        n_rows = int(np.ceil(old_div(float(n_axes), n_cols)))
        if fig is None and axs is None:
            fig, axs = plt.subplots(n_rows, n_cols, subplot_kw={"aspect": 1},
                                    figsize=figsize)
        alphas = np.linspace(0, 1, 200)
        try:
            axs = axs.ravel()
        except AttributeError:
            axs = np.array([axs])
        for ax_i, var_name in enumerate(var_names):
            var_ii = self.var_names.index(var_name)
            # we have to limit ourself to the length of the simulated
            # data.
            # TODO: we are biasing this comparision because we take
            # the first values!
            # finite_obs = np.isfinite(self.data_raw[var_ii])[:self.T_sim]
            finite_obs = np.isfinite(self.data_raw[var_ii])
            if trans:
                obs = np.sort(np.copy(self.data_trans[var_ii, finite_obs]))
                # sim = np.sort(np.copy(self.sim[var_ii, finite_obs]))
                sim = np.sort(np.copy(self.sim[var_ii]))
            else:
                obs = np.sort(np.copy(self.data_raw[var_ii, finite_obs]))
                obs /= self.sum_interval[var_ii]
                # sim = np.sort(np.copy(self.sim_sea[var_ii, finite_obs]))
                sim = np.sort(np.copy(self.sim_sea[var_ii]))
            obs = np.quantile(obs, alphas)
            sim = np.quantile(sim, alphas)
            ax = axs[ax_i]
            if lines:
                ax.plot(sim, obs, color=color)
            else:
                ax.scatter(sim, obs, marker="x", color=color)
            global_min = min(obs[0], sim[0])
            global_max = max(np.nanmax(obs), sim[-1])

            if (self.sim_sea_dis is not None and
                self.var_names_dis is not None and
                    var_name in self.var_names_dis):

                obs_dis_all = np.array(self.met[var_name]
                                       )[:self.sim_sea_dis.shape[1]]
                finite_obs = np.isfinite(obs_dis_all)
                var_i_dis = self.var_names_dis.index(var_name)
                obs_dis = np.sort(np.copy(obs_dis_all[finite_obs]))
                sim_dis = np.squeeze(self.sim_sea_dis[var_i_dis, finite_obs])
                sim_dis = np.sort(np.copy(sim_dis))
                if lines:
                    ax.plot(sim_dis, obs_dis, color="black")
                else:
                    ax.scatter(sim_dis, obs_dis, marker="x", color="black")
                global_min = min(global_min, obs_dis[0], sim_dis[0])
                global_max = max(global_max, np.nanmax(obs_dis),
                                 np.nanmax(sim_dis))

            ax.plot([global_min, global_max], [global_min, global_max], "k--")
            ax.grid()
            ax.set_ylabel("observed")
            if ax_i < (n_axes - n_cols):
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("simulated")
            ax.set_title("%s %s" %
                         (conf.ygreek[var_name], conf.units[var_name]))
        # delete axs that we did not use
        if len(axs) > n_axes:
            for ax in axs[n_axes:]:
                ax.set_axis_off()
            plt.draw()
        title_str = "QQ-plots"
        fig.suptitle(title_str)

        try:
            fig.canvas.set_window_title(title_str)
        except AttributeError:
            pass
        fig.name = "qq"
        return fig, axs

    def plot_seasonal_corrs(self, var_names=None, trans=False, figsize=None,
                          fig=None, axs=None):
        if var_names is None:
            var_names = self.var_names
        elif isinstance(var_names, basestring):
            var_names = var_names,
        if fig is None and axs is None:
            fig, axs = plt.subplots(nrows=len(var_names), ncols=1,
                                    sharex=True,
                                    constrained_layout=True,
                                    figsize=figsize)
        if trans:
            obs_df = self.to_df("daily input trans", var_names=var_names)
            sim_df = self.to_df("daily output trans", var_names=var_names)
        else:
            obs_df = self.to_df("daily input", var_names=var_names)
            sim_df = self.to_df("daily output", var_names=var_names)
        for ax, var_name in zip(axs, var_names):
            corrs_obs = np.empty((12, self.K - 1), dtype=float)
            var_i = var_names.index(var_name)
            var_other_ii = [idx for idx in range(len(var_names))
                            if idx != var_i]
            for month_i, group in obs_df.groupby(obs_df.index.month):
                corrs = np.corrcoef(group.T)
                corrs_obs[month_i - 1] = corrs[var_i, var_other_ii]
            # simulated time series can have a different length
            corrs_sim = np.empty((12, self.K - 1), dtype=float)
            for month_i, group in sim_df.groupby(sim_df.index.month):
                corrs = np.corrcoef(group.T)
                corrs_sim[month_i - 1] = corrs[var_i, var_other_ii]
            var_names_other = [var_names[idx] for idx in var_other_ii]
            for corr_obs in corrs_obs.T:
                ax.plot(corr_obs, "--")
            ax.set_prop_cycle(None)
            for corr_sim, var_name_other in zip(corrs_sim.T, var_names_other):
                ax.plot(corr_sim, label=var_name_other)
            ax.set_title(var_name)
            ax.legend(loc="best")
            ax.grid(True, zorder=0)
        fig.suptitle(f"Monthly correlations {'trans' if trans else ''}")
        return fig, axs

    def plot_windrose(self, figsize=None, seasonal=False, *args, **kwds):
        try:
            ui, vi = self.var_names.index("u"), self.var_names.index("v")
        except ValueError:
            print("Can only plot wind roses if wind speed is simulated")
            return

        u = old_div(self.data_raw[ui], self.sum_interval[ui])
        v = old_div(self.data_raw[vi], self.sum_interval[vi])
        wind_dirs = avrwind.component2angle(u, v)[0]

        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()

        def rose_func_gen(times_):
            if seasonal:
                return lambda *args, **kwds: \
                    seasonal_windroses(times_, *args, **kwds)
            else:
                return lambda *args, **kwds: windrose(*args, **kwds)

        rose_func = rose_func_gen(self.times)
        figs = [rose_func(wind_dirs, fig=fig, *args, **kwds)]
        figs[0].suptitle("measured")

        if self.sim_sea is not None:
            u, v = self.sim_sea[ui], self.sim_sea[vi]
            wind_dirs = avrwind.component2angle(u, v)[0]
            if figsize is not None:
                fig = plt.figure(figsize=figsize)
            else:
                fig = plt.figure()
            figs += [rose_func(wind_dirs, fig=fig, *args, **kwds)]
            figs[-1].suptitle("simulated")

        if self.sim_sea_dis is not None:
            rose_func = rose_func_gen(self.times_orig)
            u, v = self.met["u"], self.met["v"]
            wind_dirs = avrwind.component2angle(u, v)[0]
            if figsize is not None:
                fig = plt.figure(figsize=figsize)
            else:
                fig = plt.figure()
            figs += [rose_func(wind_dirs, fig=fig, *args, **kwds)]
            figs[-1].suptitle("measured hourly")

            rose_func = rose_func_gen(self.dis_times)
            ui_dis = self.var_names_dis.index("u")
            vi_dis = self.var_names_dis.index("v")
            u, v = self.sim_sea_dis[ui_dis], self.sim_sea_dis[vi_dis]
            wind_dirs = avrwind.component2angle(u, v)[0]
            if figsize is not None:
                fig = plt.figure(figsize=figsize)
            else:
                fig = plt.figure()
            figs += [rose_func(wind_dirs, fig=fig, *args, **kwds)]
            figs[-1].suptitle("simulated disaggregated")

        return np.squeeze(figs)

    def plot_corr(self, ygreek=None, hourly=False, trans=False, *args, **kwds):
        if ygreek is None:
            # get rid of the incident for this block
            ygreek = collections.defaultdict(lambda: r"-")
            ygreek.update({"R":      r'$R$',
                           "theta":  r'$\theta$',
                           "rh":     r"$\phi$",
                           "Qsw":    r"$Q_{sw}$",
                           "ILWR":   r"$Q_{lw(i.)}$",
                           "u":      r"$u$",
                           "v":      r"$v$",
                           })

        if self.ex_in is None:
            data = self.data_raw
            var_names = self.var_names
        else:
            data = np.vstack((self.data_raw, self.ex_in))
            var_names = list(self.var_names) + ["external"]

        def my_greek(var_names):
            return [ygreek[var_name] for var_name in var_names]

        greek_short = my_greek(var_names)
        figs, axs = append_fa(ts.corr_img(data, 0, "Measured daily",
                                          greek_short, *args, **kwds))
        figs[0].name = "corr_measured_daily"
        
        if trans:
            figs, axs = append_fa(ts.corr_img(self.data_trans, 0,
                                              "Measured daily transformed",
                                              greek_short, *args, **kwds),
                                  figs, axs)
            figs[-1].name = "corr_measured_daily_trans"

        if self.sim_sea is not None:
            if self.ex_out is None:
                data = self.sim_sea
                var_names = self.var_names
            else:
                data = np.vstack((self.sim_sea, self.ex_out))
                var_names = list(self.var_names) + ["external"]
            greek_short = my_greek(var_names)
            figs, axs = append_fa(ts.corr_img(data, 0, "Simulated daily",
                                              greek_short, *args, **kwds),
                                  figs, axs)
            figs[-1].name = "corr_sim_daily"
            if trans:
                figs, axs = append_fa(ts.corr_img(self.sim, 0,
                                                  "Simulated daily transformed",
                                                  greek_short, *args, **kwds),
                                      figs, axs)
                figs[-1].name = "corr_sim_daily_trans"

        if hourly:
            met_array = vg_base.met_as_array(self.met,
                                             var_names=self.var_names)
            figs, axs = append_fa(ts.corr_img(met_array, 0, "Measured hourly",
                                              greek_short, *args, **kwds),
                                  figs, axs)
            figs[-1].name = "corr_measured_hourly"

            if self.sim_sea_dis is not None:
                figs, axs = append_fa(ts.corr_img(self.sim_sea_dis, 0,
                                                  "Simulated hourly",
                                                  greek_short, *args, **kwds),
                                      figs, axs)
                figs[-1].name = "corr_sim_hourly"
        return figs, axs

    def plot_VAR_par(self, **kwds):
        """Plots the parameters of the VAR-Process."""
        if self.AM is None:
            print("Call fit first")
            return
        return ts.matr_img(np.asarray(self.AM),
                           "AM p=%d q=%d" % (self.p, self.q), None,
                           self.var_names, **kwds)

    def plot_autocorr(self, maxlag=7, title=None, var_names=None, *args,
                      **kwds):
        """Plots autocorrelation of transformed and simulated data (if the
        latter is available).

        Parameters
        ----------
        maxlag : int, optional
            Maximum number of lags to plot
        """
        data = self.data_trans
        figs = []
        if self.sim is not None:
            data = [data, self.sim]
        if title is None:
            title_ = "Autocorrelation of standard-normal distributed variables"
        if var_names is None:
            var_names = var_names_greek(self.var_names)
        figs, axs = append_fa(ts.plot_auto_corr(data, maxlag, title_,
                                                var_names, *args, **kwds))
        data = self.data_raw
        if self.sim_sea is not None:
            data = [data, self.sim_sea]
        if title is None:
            title_ = "Autocorrelation of untransformed variables"
        figs, axs = append_fa(ts.plot_auto_corr(data, maxlag, title_,
                                                var_names, *args, **kwds),
                              figs, axs)
        if self.residuals is not None:
            figs, axs = append_fa(ts.plot_auto_corr(self.residuals, maxlag,
                                                     "Autocorrelation of residuals.",
                                                     var_names, **kwds),
                                  figs, axs)
            # otherwise we probably have been given two arrays.  then
            # things get more complicated, because we would distinguish
            # between two possibly different sample sizes (data.shape[1])
            if isinstance(data, np.ndarray):
                # bounds for test on the whiteness of the residuals (see page 160)
                bound = 2. / np.sqrt(data.shape[1]) * np.ones(maxlag)
                plt.plot(list(range(maxlag)), bound, "--", color="grey")
                plt.plot(list(range(maxlag)), -bound, "--", color="grey")

            if self.ut is None:
                squared_res = self.residuals ** 2
            else:
                squared_res = [self.residuals ** 2,
                               self.ut ** 2]
            figs, axs = append_fa(ts.plot_auto_corr(squared_res, maxlag,
                                                    "Autocorrelation of "
                                                    "squared residuals.",
                                                    var_names, **kwds),
                                  figs, axs)
        return figs, axs

    def plot_cross_corr(self, max_lags=7, figsize=None):
        """Plot cross correlations."""
        kwds = dict(var_names=self.var_names, max_lags=max_lags)
        fig, axs = ts.plot_cross_corr(old_div(self.data_raw, self.sum_interval),
                                      figsize=figsize, **kwds)
        if self.sim_sea is not None:
            fig, axs = ts.plot_cross_corr(self.sim_sea, linestyle="--",
                                          fig=fig, axs=axs, **kwds)
        suptitle = "Daily"
        plt.suptitle(suptitle)

        if self.sim_sea_dis is not None:
            met_array = vg_base.met_as_array(self.met,
                                             var_names=self.var_names)
            kwds["max_lags"] *= max(self.sum_interval)
            fig1, axs1 = ts.plot_cross_corr(met_array, figsize=figsize, **kwds)
            fig1, axs1 = ts.plot_cross_corr(self.sim_sea_dis, linestyle="--",
                                            fig=fig1, axs=axs1, **kwds)
            suptitle = "Hourly"
            plt.suptitle(suptitle)
            fig = [fig, fig1]
            axs = [axs, axs1]

        return fig, axs

    def plot_scaling(self, agg_funcs=(np.mean, np.std, stats.skew,
                                      stats.kurtosis),
                     max_agg_len=2 * 365, normalize=True, **fig_kw):
        """Plot mean/std/skew over aggregation length."""
        kwds = dict(var_names=self.var_names, agg_funcs=agg_funcs,
                    max_agg_len=max_agg_len, **fig_kw)
        fig, axs = ts.plot_scaling(self.data_raw / self.sum_interval,
                                   color="black", **kwds)

        if self.sim_sea is not None:
            sim = self.sim_sea
            if normalize:
                # for the sake of argument: make the means equal
                obs_means = np.nanmean(self.data_raw / self.sum_interval,
                                       axis=1)
                sim_means = np.nanmean(self.sim_sea, axis=1)
                sim += (obs_means - sim_means)[:, None]
            fig, axs = ts.plot_scaling(sim, color="blue", fig=fig, axs=axs,
                                       **kwds)
        return fig, axs

    def plot_monthly_hists(self, var_names=None, bins=20, figs=None, axss=None):
        """Plot histograms grouped by month and variable."""
        if var_names is None:
            var_names = self.var_names
        elif isinstance(var_names, basestring):
            var_names = var_names,

        if figs is None:
            figs = []
            reuse_figs = False
        else:
            reuse_figs = True
        data = self.data_raw / self.sum_interval
        month_ii = [times.time_part(self.times, "%m") == month
                    for month in range(1, 13)]
        if self.sim_sea is not None:
            sim_month_ii = [times.time_part(self.sim_times, "%m") == month
                            for month in range(1, 13)]
        for var_name in var_names:
            var_i = self.var_names.index(var_name)
            if reuse_figs:
                fig, axs = figs[var_i], axs[var_i]
            else:
                fig, axs = plt.subplots(3, 4, sharey=True, sharex=True)
                axs = axs.ravel()
            for month in range(12):
                dat = data[var_i, month_ii[month]]
                dat = dat[np.isfinite(dat)]
                label = "obs"
                if self.sim_sea is not None:
                    label = [label, "sim"]
                    dat = [dat, self.sim_sea[var_i, sim_month_ii[month]]]
                axs[month].hist(dat, bins, density=True, histtype="step",
                                label=label)
                if self.sim_sea is None:
                    dat = dat,
                ax_cdf = axs[month].twinx()
                for values in dat:
                    values = values[np.isfinite(values)]
                    ranks = (np.arange(len(values)) - .5) / len(values)
                    ax_cdf.plot(np.sort(values), ranks)
                    ax_cdf.set_yticklabels([])
                ax_cdf.set_ylim(0, 1)
                axs[month].set_title(month + 1)
            if self.sim_sea is not None:
                axs[0].legend(loc="best")

            suptitle = var_name
            fig.suptitle(suptitle)
            fig.tight_layout(rect=(0, 0, 1, .95))
            if not reuse_figs:
                figs += [fig]
        if len(figs) == 1 and not reuse_figs:
            figs = figs[0]
        for ax in axs:
            ax.grid(True)
        return figs, axs

    def plot_monthly_hists_hourly(self, var_names=None, bins=20):
        """Plot histograms grouped by month and variable."""
        if var_names is None:
            var_names = self.var_names
        elif isinstance(var_names, basestring):
            var_names = var_names,

        figs = []
        data = vg_base.met_as_array(self.met, var_names=var_names)
        month_ii = [times.time_part(self.times_orig, "%m") == month
                    for month in range(1, 13)]
        if self.sim_sea_dis is not None:
            sim_month_ii = [times.time_part(self.dis_times, "%m") == month
                            for month in range(1, 13)]
        for var_name in var_names:
            var_i = self.var_names.index(var_name)
            fig, axs = plt.subplots(3, 4, sharey=True, sharex=True)
            axs = axs.ravel()
            for month in range(12):
                dat = data[var_i, month_ii[month]]
                dat = dat[np.isfinite(dat)]
                label = "obs"
                if self.sim_sea is not None:
                    label = [label, "sim"]
                    dat = [dat, self.sim_sea_dis[var_i, sim_month_ii[month]]]
                axs[month].hist(dat, bins, density=True, histtype="step",
                                label=label)
                if self.sim_sea is None:
                    dat = dat,
                ax_cdf = axs[month].twinx()
                for values in dat:
                    values = values[np.isfinite(values)]
                    ranks = (np.arange(len(values)) - .5) / len(values)
                    ax_cdf.plot(np.sort(values), ranks)
                    ax_cdf.set_yticklabels([])
                ax_cdf.set_ylim(0, 1)
                axs[month].set_title(month + 1)
            if self.sim_sea is not None:
                axs[0].legend(loc="best")

            suptitle = var_name
            fig.suptitle(suptitle)
            fig.tight_layout(rect=(0, 0, 1, .95))
            figs += [fig]
        if len(figs) == 1:
            figs = figs[0]
        for ax in axs:
            ax.grid(True)
        return figs, axs

    def plot_episode_hists(self, var_names=None):
        """Plots histograms of episode duration and amplitude grouped by
        variable."""
        if var_names is None:
            var_names = self.var_names
        elif isinstance(var_names, basestring):
            var_names = var_names,

        figs = []
        axs = []
        for var_name in var_names:
            var_i = self.var_names.index(var_name)
            medians = self.fitted_medians(var_name, self.data_doys)
            devs_obs = \
                self.data_raw[var_i] / self.sum_interval[var_i] - medians
            sign_changes = np.where(np.diff(np.sign(devs_obs)) != 0)[0]
            durations = [np.diff(sign_changes)]
            amplitudes = [np.array([np.mean(epi) for epi in
                                    np.split(devs_obs, sign_changes)
                                    if len(epi) > 0])]

            if self.sim_sea is not None:
                medians = self.fitted_medians(var_name, self.sim_doys)
                devs_sim = self.sim_sea[var_i] - medians
                sign_changes = np.where(np.diff(np.sign(devs_sim)) != 0)[0]
                durations_sim = np.diff(sign_changes)
                amplitudes_sim = [np.mean(epi) for epi in
                                  np.split(devs_sim, sign_changes)
                                  if len(epi) > 0]
                durations += [durations_sim]
                amplitudes += [amplitudes_sim]

            fig, axs_ = plt.subplots(2, 1)
            axs_[0].hist(durations, 30, density=True)
            axs_[0].set_xlabel("Duration [days]")
            axs_[1].hist(amplitudes, 30, density=True)
            axs_[1].set_xlabel("Deviation %s" % conf.units[var_name])

            legend1 = (r"observed $\overline{x}=%.2f$" %
                       np.mean(durations[0] - 1))
            legend2 = (r"observed $\sigma=%.2f$" %
                       np.std(amplitudes[0][np.isfinite(amplitudes[0])]))
            if self.sim_sea is not None:
                legend1 = [legend1, r"sim $\overline{x}=%.2f$" %
                           np.mean(durations_sim - 1)]
                legend2 = [legend2, r"sim $\sigma=%.2f$" %
                           np.std(amplitudes_sim)]
            else:
                legend1, legend2 = (legend1,), (legend2,)
            axs_[0].legend(legend1, loc="best")
            axs_[1].legend(legend2, loc="best")

            suptitle = "Episode statistics %s" % var_name
            fig.suptitle(suptitle)
            try:
                fig.canvas.set_window_title(suptitle)
            except AttributeError:
                pass
            figs += [fig]
            axs += [axs_]
        return figs, axs

    def plot_daily_cycles(self, var_names=None, rain_thresh=None, **fig_kw):
        """Plot cycle of hourly means over the year.

        Climbing Mt. Allon.

        Parameter
        ---------
        var_names : sequence of str, optional
            Which variables to plot. None means plot all.
        rain_thresh : None, True or float, optional
            Which elemts to include based on rain threshold. If True, it is
            inferred from the configuration file (dist_kwds["R"]["threshold"])
        """
        if self.sim_sea_dis is None:
            print("Call disaggregate first")
            return
        if var_names is None:
            var_names = self.var_names
        elif type(var_names) == str:
            var_names = var_names,

        if not rain_thresh:
            # rain_mask_obs = np.ones_like(self.met["R"], dtype=bool)
            # rain_mask_sim = np.ones(self.sim_sea_dis.shape[1], dtype=bool)
            rain_comparison = False
        if rain_thresh is True or type(rain_thresh) is float:
            if rain_thresh is True:
                rain_thresh = conf.threshold
            rain_mask_obs = (self.data_raw[self.var_names.index("R")] >=
                             rain_thresh)
            rain_mask_obs = np.repeat(rain_mask_obs,
                                      self.sum_interval_dict["R"][0])
            rain_mask_sim = (self.sim_sea[self.var_names.index("R")] >=
                             rain_thresh)
            rain_mask_sim = np.repeat(rain_mask_sim,
                                      self.sum_interval_dict["R"][0])
            rain_comparison = True

        figs, axes = [], []
        month_iis = [times.time_part(self.times_orig, "%m") == month
                     for month in range(1, 13)]
        hour_iis = [times.time_part(self.times_orig, "%H") == hour
                    for hour in range(24)]
        month_sim_iis = [times.time_part(self.dis_times, "%m") == month
                         for month in range(1, 13)]
        hour_sim_iis = [times.time_part(self.dis_times, "%H") == hour
                        for hour in range(24)]

        fig_kw.update(dict(sharex=True, sharey=True))
        for var_name in var_names:
            if rain_comparison:
                fig, axs = plt.subplots(2, 2, **fig_kw)
            else:
                fig, axs = plt.subplots(1, 2, **fig_kw)
                axs = np.array((axs,))

            var_i_dis = self.var_names.index(var_name)

            for mask_i, mask_mod in enumerate((lambda x: x, lambda x: ~x)):
                if not rain_comparison and mask_i == 1:
                    break

                means_obs = np.zeros((12, 24))
                means_sim = np.zeros_like(means_obs)
                for month in range(1, 13):
                    for hour in range(24):
                        month_ii = month_iis[month - 1]
                        hour_ii = hour_iis[hour]
                        mask = month_ii & hour_ii
                        if rain_comparison:
                            mask &= mask_mod(rain_mask_obs)
                        data_obs = np.array(self.met[var_name])[mask]
                        month_ii = month_sim_iis[month - 1]
                        hour_ii = hour_sim_iis[hour]
                        data_sim = self.sim_sea_dis[var_i_dis]
                        mask = month_ii & hour_ii
                        if rain_comparison:
                            mask &= mask_mod(rain_mask_sim)
                        data_sim = data_sim[mask[:len(data_sim)]]
                        means_obs[month - 1, hour] = np.nanmean(data_obs)
                        means_sim[month - 1, hour] = np.nanmean(data_sim)

                # the following code tries to achieve that the colorbar for the
                # two contour plots spans from vmin to vmax (this would be the
                # case for calls to imshow instead of contourf).
                # see
                # http://matplotlib.1069221.n5.nabble.com/
                # question-about-contours-and-clim-td21111.html
                # and
                # https://github.com/matplotlib/matplotlib/pull/2176
                # to understand why this is so annoying
                vmin = min(means_obs.min(), means_sim.min())
                vmax = max(means_obs.max(), means_sim.max())
                locator = ticker.MaxNLocator(8)
                locator.create_dummy_axis()
                locator.set_bounds(vmin, vmax)
                levs = locator()
                axs[mask_i, 0].contourf(list(range(1, 25)),
                                        list(range(1, 13)),
                                        means_obs, levs)
                co = axs[mask_i, 1].contourf(list(range(1, 25)),
                                             list(range(1, 13)),
                                             means_sim, levs)

                axs[mask_i, 0].set_xticks([6, 12, 18, 24])
                axs[mask_i, 1].set_xticks([6, 12, 18, 24])

                axs[mask_i, 0].set_title("Observed")
                axs[mask_i, 0].set_xlabel("hour")
                axs[mask_i, 0].set_ylabel("month")
                axs[mask_i, 0].grid()
                axs[mask_i, 1].set_title("Simulated")
                axs[mask_i, 1].set_xlabel("hour")
                axs[mask_i, 1].grid()

            title = "Seasonal daily cycles for %s" % conf.ygreek[var_name]
            fig.suptitle(title)
            fig.name = "daily_cycles_%s" % var_name
            try:
                fig.canvas.set_window_title(title)
            except AttributeError:
                pass
            if "constrained_layout" not in fig_kw.keys():
                fig.tight_layout(rect=(0, 0, 1, .925))
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(co, cax=cbar_ax)
            cbar.set_label(conf.units[var_name])

            figs += [fig]
            axes += [axs]
        return figs, axes

    # def plot_torus(self, var_names=None, **fig_kw):
    #     """Provides a 2d visualization with doys as horizontal and hours as
    #     vertical axis.

    #     Parameter
    #     ---------
    #     var_names : sequence of str, optional
    #         Which variables to plot. None means plot all.
    #     """
    #     if var_names is None:
    #         var_names = self.var_names_dis
    #     elif type(var_names) == str:
    #         var_names = var_names,

    #     figs, axes = [], []
    #     n_years = self.times_orig[-1].year - self.times_orig[0].year + 1
    #     nrows = int(np.ceil(float(n_years) / 10))
    #     ncols = n_years / nrows
    #     for var_name in var_names:
    #         fig, axs = plt.subplots(nrows, ncols, **fig_kw)
    #         # torus = 
            
    #         figs += [fig]
    #         axes += [axs]
