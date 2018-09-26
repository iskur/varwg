"""
Helper functions for Monte Carlo simulations (:mod:`vg.core.monty`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Higher-level helper functions to ease generation of input files for
DYRESM-CAEDYM and ELCOM.

.. currentmodule:: vg.core.monty

.. autosummary::
   :nosignatures:
   :toctree: generated/

    main
    vg2glm
    vg2dyresm
    vg2elcom
    vg_for_elcom
"""
from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
try:
    from importlib import reload
except ImportError:
    pass
import os
import shutil
import glob
import time
import warnings
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats as sp_stats
try:
    from ipyparallel import Client
except ImportError:
    from IPython.parallel import Client

import vg
from vg import helpers as my
from vg.smoothing import smooth
from vg import times
from vg.meteo import meteox2y as mxy
from vg.meteo import avrwind, windrose


def main(nn, pth, varnames=("Qsw", "ILWR", "theta", "rh", "u", "v"),
         n1=0, varnames_dis=("Qsw", "u", "v"), wind_fkt=1.3, p=3, **kwargs):
    """
    Generate many realisations of meteorological input data for
    Monty-Python (also known as Monte-Carlo) simulations with DYRESM-CAEDYM

    Parameters
    ----------
    nn : int
        number of realisations
    pth : string
        path to folder to store the nn files
    varnames : tuple of str, optional
        variables to simulate
    n1 : int, optional
        running index to start from
    varnames_dis : list of str, optional
        variables to disaggregate to hourly values
    wind_fkt : float, optional
        factor for wind speed. Default: 1.3 (for Konstanz wind data)
    p : int, optional
        Autoregressive order of the VAR-process. (number of previous days to
        consider).
        3 was suitable for a DWD-Constance dataset.
    **kwargs:
        vg parameters
    """
    met_vg = vg.VG(varnames, plot=False)
    met_vg.fit(p=p)
    p = met_vg.p
    for n in range(n1, nn):
        print('\n*************************')
        times_, sim = met_vg.simulate(**kwargs)
        # simulate sometimes produces NANs. We don't want them. So if there are
        # any, try again:
        while ~np.isfinite(np.average(sim[2, :])):
            times_, sim = met_vg.simulate(**kwargs)
        if varnames_dis:
            times_, sim = met_vg.disaggregate(varnames_dis)
        if q_stuff:
            if n == n1:
                data = sim
            else:
                data = np.append(data, sim)
        # make dictionary:
        sim = np.append(sim, times_).reshape(len(varnames) + 1, -1)
        varnames__ = varnames + ("time",)
        sim_ = dict((varn.strip(), simi)
                    for varn, simi in zip(varnames__, sim))
        #
        metfilename = 'vg_%04d.met' % (n)
        if not os.path.exists(pth):
            os.mkdir(pth)
        met = os.path.join(pth, metfilename)
        print(n, met)
        vg2dyresm(sim_, met, teiler=1.0, ts=3600, info=str(kwargs),
                  wind_fkt=wind_fkt)


class Monty(object):
    """Class to run any number of VG realizations in parallel."""
    def __init__(self, n_processes=None, conf=None):
        """Starts up ipcluster. Implements the Context-Manager protocol (i.e.
        use it with a "with"-statement).

        Parameters
        ----------
        n_processes : int or None, optional
            Number of parallel processes to be run. If None, ipcluster will
            decide (that will be number of threads or cores).
        """
        self.n_processes = n_processes
        self.conf = vg.conf if conf is None else conf

    def __enter__(self):
        command = ["ipcluster", "start", "local",
                   # "--debug",
                   "--profile=default"]
        if self.n_processes is not None:
            command += ["-n", str(self.n_processes)]
        subprocess.Popen(command)
        # we have to wait a bit until the engines are hot. there might be a
        # less clumsy way than waiting for a fixed amount of time
        time.sleep(10)
        client = Client()
        # direct view (vs. load-balanced)
        self.dview = client[:]
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        subprocess.Popen(["ipcluster", "stop"])

    def init(self, vg_kwds, fit_kwds=None):
        """Initializes a VG object and calls fit on it.

        Parameters
        ----------
        vg_kwds : dict
            Keyword-arguments to be passed to VG.__init__.
            dump_data will be set to False.
        fit_kwds : dict or None, optional
            Keyword-arguments to be passed to VG.fit.
        """
        if fit_kwds is None:
            fit_kwds = {}
        self.dview.apply_sync(_init, self.conf.__name__, vg_kwds, fit_kwds)
        self.dview.execute("import numpy as np")

    def run(self, n_realizations=100, use_seed=True, sim_kwds=None,
            dis_kwds=None, dyresm_kwds=None, glm_kwds=None):
        """Run simulate, disaggregate, and to_dyresm and/or to_glm.

        Parameters
        ----------
        n_realizations : int
            Number of realizations to run.
        use_seed : boolean, optional
            Seed every simulation with realization index.
        sim_kwds : dict or None, optional
            Keyword-arguments to be passed to VG.simulate.
        dis_kwds : dict or None, optional
            Keyword-arguments to be passed to VG.disaggregate.
        dyresm_kwds : dict or None, optional
            Keyword-arguments to be passed to VG.to_dyresm.
            If None, no dyresm output will be generated.
        glm_kwds : dict or None, optional
            Keyword-arguments to be passed to VG.to_dyresm.
            If None, no dyresm output will be generated.
        """
        params = {key: val for key, val in list(locals().items())
                  if key not in ("self", "n_realizations")}
        tasks = [[ii, params] for ii in range(n_realizations)]
        self.dview.map_sync(_run_one_realization, tasks)


def _init(vg_config_name, vg_kwds, fit_kwds):
    """This is run on a cluster node."""
    import importlib
    import vg
    # try:
    #     from importlib import reload
    # except ImportError:
    #     pass
    import sys
    PY2 = sys.version_info.major == 2
    if not PY2:
        reload = importlib.reload
    else:
        importlib.reload = reload
    vg.conf = importlib.import_module(vg_config_name)
    importlib.reload(vg.conf)
    vg_kwds["dump_data"] = False
    met_vg = vg.VG(**vg_kwds)
    met_vg.fit(**fit_kwds)
    _run_one_realization.met_vg = met_vg


def _run_one_realization(task):
    """This is run on a cluster node. Configuration parameters where pushed
    here from Monty, but a some modules have to be imported to make this work.
    """
    ri, params = task
    met_vg = _run_one_realization.met_vg

    if params["use_seed"]:
        np.random.seed(ri)

    sim_kwds = params["sim_kwds"]
    if sim_kwds is None:
        sim_kwds = {}
    met_vg.simulate(**sim_kwds)

    dis_kwds = params["dis_kwds"]
    if dis_kwds is None:
        dis_kwds = {}
    met_vg.disaggregate(**dis_kwds)

    glm_kwds = params["glm_kwds"]
    if glm_kwds:
        outfilepath = glm_kwds["outfilepath"] % ri
        kwds = {key: val for key, val in list(glm_kwds.items())
                if key != "outfilepath"}
        if outfilepath:
            met_vg.to_glm(outfilepath, **kwds)

    dyresm_kwds = params["dyresm_kwds"]
    if dyresm_kwds:
        outfilepath = dyresm_kwds["outfilepath"] % ri
        kwds = {key: val for key, val in list(dyresm_kwds.items())
                if key != "outfilepath"}
        if outfilepath:
            met_vg.to_dyresm(outfilepath, **kwds)


def vg_for_elcom(pth, varnames=("Qsw", "ILWR", "theta", "rh", "u", "v"),
                metfilename='meteo_vg.d', windfilename='input_wind_vg.d',
                plotting=False, varnames_dis=("Qsw", "u", "v"), p=3,
                **kwargs):
    """Generate meteorology using VG and make ELCOM input wind and meteo
    boundary condition files.

    Parameters
    ----------
    pth : string
        path to folder to store the elcom files
    varnames : tuple of str, optional
        variables to simulate
    metfilename : string, optional
    windfilename : string, optional
    plotting : boolean, optional
    varnames_dis : list of str, optional
        variables to disaggregate to hourly values
    p : int, optional
        Autoregressive order of the VAR-process. (number of previous days to
        consider).
        3 was suitable for a DWD-Constance dataset.
    **kwargs:
        vg parameters
    """
    met_vg = vg.VG(varnames, plot=False)
    met_vg.fit(p=p)
    times_, sim = met_vg.simulate(**kwargs)
    # simulate sometimes produces NANs. We don't want them. So if there are
    # any, try again:
    while ~np.isfinite(np.average(sim[2, :])):
        times_, sim = met_vg.simulate(**kwargs)
    if varnames_dis:
        times_, sim = met_vg.disaggregate(varnames_dis)
    # make dictionary:
    sim = np.append(sim, times_).reshape(len(varnames) + 1, -1)
    varnames = varnames + ("time",)
    sim_ = dict((varn.strip(), simi) for varn, simi in zip(varnames, sim))
    met = os.path.join(pth, metfilename)
    wind = os.path.join(pth, windfilename)
    vg2elcom(sim_, met, teiler=1.0, ts=3600, wind=wind, windfaktor=1.3,
                simdict=True, info=str(kwargs))
    # save random_state:
    p = met_vg.p
    if met_vg.q == None:
        q = 0
    else:
        q = met_vg.q
    random_state = os.path.join(vg.conf.data_dir,
                                'VARMA_p%i_q%i_sim.random_state' % (p, q))
    random_save = os.path.join(pth, '%s_p%i_q%i.random_state'
                               % (metfilename[:-2], p, q))
    shutil.copy(random_state, random_save)
    if plotting:
        print('plotting')
        plot_elcom_meteo(sim_, windfile=wind, simdict=True)
        figname = os.path.join(pth, metfilename[:-2] + '.png')
        plt.savefig(figname)


def vg2glm(vg_filename, glm_filename):
    """Convert VG data to GLM input.

    Parameters
    ----------
    vg_out_filename : path/filename
        vg output (txt file)
    glm_in_filename : path/filename
        glm meteorological bc file
    """
    header = "time,ShortWave,LongWave, AirTemp,RelHum, WindSpeed,Rain,Snow\n"
    dtimes, met = vg.read_met(vg_filename, verbose=False)
    if "U" not in met:
        try:
            met["U"] = avrwind.component2angle(met["u"], met["v"])[1]
        except KeyError:
            warnings.warn("No wind information found. Filling in 0's!!")
            met["U"] = np.zeros_like(met[list(met.keys())[0]])
    times_str = times.datetime2str(dtimes, "%Y-%m-%d %H:%M")
    lines = [",".join([time_str] +
                      ["%.6f" % met[key][i]
                       for key in ("Qsw", "ILWR", "theta", "rh", "U", "R")] +
                      ["0.0\n"]  # snow
                      )
             for i, time_str in enumerate(times_str)]
    with open(glm_filename, "w") as glm_file:
        glm_file.write(header)
        glm_file.writelines(lines)


def vg2dyresm(meteo, met, ts=None, info=None, wind_fkt=1.0, output_rh=False,
              verbose=False):
    """ write VG data to DYRESM input meteo boundary condition txt file

    Parameters
    ----------
    meteo : path/filename or dictionary
        vg output (txt file)
    met : path/filename
        dyresm meteorological bc file
    ts : int
        timestep in dyresm-met-file in seconds, should be divisor of 86400
    info : information text string in file header
    wind_fkt : float
        factor for wind speed. Default: 1.0 (for Konstanz wind data: 1.3)
    """
    if type(meteo) == str:
#         alt_text = meteo
        meteo = my.csv2dict(meteo, delimiter='\t')
        # date = times.iso2unix(meteo["time"])
        date = times.iso2datetime(meteo["time"])
    else:
#         alt_text = 'vg_sim_data'
        # date = times.datetime2unix(meteo["time"])
        date = meteo["time"]
    if ts is None:
        # ts = int(date[1] - date[0])
        ts = (date[1] - date[0]).total_seconds()
    meteo_dy = open(met, 'w')
    sw = np.array(meteo["Qsw"], dtype=float)
    sw[np.where(sw < 0)] = 0
    lw = np.array(meteo["ILWR"], dtype=float)
    at = np.array(meteo["theta"], dtype=float)
    try:
        U = np.array(meteo["U"], dtype=float)
        U[np.where(U <= 0)] = 0
    except KeyError:
        u = np.array(meteo["u"], dtype=float)
        v = np.array(meteo["v"], dtype=float)
        if verbose:
            print('wind was given in components')
        U = avrwind.component2angle(u, v)[1]
    U = wind_fkt * U
    if output_rh:
        rh = np.array(meteo["rh"], dtype=float)
    else:
        try:
            e = np.array(meteo["e"], dtype=float)
        except KeyError:
            rh = np.array(meteo["rh"], dtype=float)
            e = mxy.rel2vap_p(rh, at)
    try:
        r = np.array(meteo["R"], dtype=float)
    except KeyError:
        r = np.zeros_like(at)  # it never rains in southern vg
    zeit = time.strftime("%d. %B %Y %H:%M:%S", time.localtime())
    meteo_dy.write('<#3>\nComment line: generated by monty on %s with %s\n'
                                % (zeit, info))
    meteo_dy.write('%i # met data input time step (seconds)\n' % ts)
    meteo_dy.write('INCIDENT_LW #longwave radiation type (NETT LW, '
                   'INCIDENT LW,CLOUD COVER)\n')
    meteo_dy.write('FLOATING 10 # sensor type (FLOATING; FIXED HT), '
                   'height in m (above water surface; above lake bottom)\n')
    meteo_dy.write('Julian	Qsw	ILWR	theta	%s	U	R\n' %
                   ("rh" if output_rh else "e"))
    # date = times.unix2cwr(date)
    date = times.datetime2cwr(date)
    data = np.array((date, sw, lw, at, rh if output_rh else e, U, r)
                    ).transpose(1, 0)
    np.savetxt(meteo_dy, data,
        fmt=('%10.5f\t%7.3f\t%7.3f\t%6.2f\t' +
             ("%.3f" if output_rh else "%4.1f") +
             '\t%5.2f\t%.6f'))
    meteo_dy.close()


def test_theta_incr(theta_incr_max=15., delta=0.5, typ='incr',
                    varnames=("theta", "Qsw", "ILWR", "rh", "U"), nn=1):
    """
    theta should be the first in varnames """
    met_vg = vg.VG(varnames, plot=False)
    met_vg.fit(p=2, q=3)
    th_i = np.arange(delta, theta_incr_max + delta, delta).repeat(nn)
    fkt, temp_end, nans = [], [], []
    for theta_incr in th_i:
        print('\n*************************')
        if typ == 'incr':
            sim = met_vg.simulate(T=None, mean_arrival=7,
                            disturbance_std=0.5, theta_incr=theta_incr,
                            theta_grad=0)[1]
        elif typ == 'grad':
            sim = met_vg.simulate(T=None, mean_arrival=7,
                            disturbance_std=0.5, theta_incr=0.,
                            theta_grad=theta_incr)[1]
        # replace inf by nan:
        sim[0, np.where(np.isinf(sim[0, :]))] = np.nan
        av_incr = np.nanmean(sim[0, :]) - 9.5755
        fkt.append(old_div(av_incr, theta_incr))
        print('type of increase: ', typ)
        print('theta_incr =', theta_incr)
        print('average temperature increase =', av_incr)
        print('factor = ', fkt[-1])
        nans.append(np.sum(np.isnan(sim[0, :])))
        if typ == 'grad':
            temp_end.append(np.nanmean(sim[0, -365:]) - 9.5755)
            print('average temperature last year:', temp_end[-1])
    plt.figure()
    plt.scatter(th_i, fkt, c=nans)
    if typ == 'grad':
        plt.scatter(th_i, old_div(temp_end, th_i), c=nans, marker='^')
    plt.grid()
    plt.xlabel('theta_incr (user defined) [$^{\circ}$C]')
    plt.ylabel('theta_incr (out) / theta_incr (user)')


def q_stuff(pth, plotting=True):
    for i, infile in enumerate(glob.glob(os.path.join(pth, '*.met'))):
##        print "current file is: " + infile
        print(i, end=' ')
        if i == 0:
            ifile = open(infile, "r")
            for _ in range(6):
                line = ifile.readline()
            varnames = line.split('\t')[1:]
            ifile.close()
            print(varnames)
            data = np.loadtxt(infile, skiprows=6,
                              converters={0: times.cwr2unix})
            date = times.unix2datetime(data[:, 0])[::24]
            data = np.average(data[:, 1:].reshape(-1, 24, len(varnames)),
                              axis=1)
        else:
            dat = np.loadtxt(infile, skiprows=6)
            dat = dat[:, 1:].reshape(-1, 24, len(varnames))
            data = np.append(data, np.average(dat, axis=1))
    print(data.shape)
    data = data.reshape(i + 1, -1, len(varnames))
    if plotting:
        for k in range(len(varnames)):
            print(k, varnames[k])
            median = np.median(data[:, :, k], axis=0)
            q_10 = sp_stats.scoreatpercentile(data[:, :, k], 10)
            q_90 = sp_stats.scoreatpercentile(data[:, :, k], 90)
            # if we plot from more than one directory in a session a k would
            # plot all scenarios in one figure per variable
            plt.figure(figsize=(18, 6))
            maxi = np.max(data[:, :, k], axis=0)
            mini = np.min(data[:, :, k], axis=0)
            plt.plot(date, maxi, 'k--', label='max')
            plt.plot(date, mini, 'k:', label='min')
            plt.plot(date, data[0, :, k], c=[.2, .2, .2], alpha=0.5)
            plt.plot(date, data[old_div(i, 2), :, k], c=[.5, .5, .5], alpha=0.5)
            plt.plot(date, data[i - 1, :, k], c=[.8, .8, .8], alpha=0.5)
            plt.plot(date, median, label='median', linewidth=2)
            plt.plot(date, q_10, label='q10', linewidth=2)
            plt.plot(date, q_90, label='q90', linewidth=2)
            plt.legend()
            plt.grid('on')
            plt.title(varnames[k])
            picfilename = os.path.join(pth, 'vg_%s.png' % (varnames[k]))
            plt.savefig(picfilename)
    else:
        return date, data


def vg2elcom(meteo, met, wind=None, ts=86400, windfaktor=1.3, simdict=True,
             info=True):
    """Write VG data to ELCOM input meteo and wind boundary condition txt files

    Parameters
    ----------
    meteo : path/filename or dictionary
        vg output (txt file)
    met : path/filename
        elcom meteorological bc file
    wind : None or path/filename
        elcom wind bc file
    teiler : float
        Dirk liefert manchmal Tagessummen der Stundenwerte -> durch 24 teilen
    windfaktor : float
        Windgeschwindigkeit wird damit multipliziert
    simdict : boole
        meteo is dictionary
    info : information text string in file header
    """
    if simdict == False:
        print('inputfile:', meteo)
        meteo = my.csv2dict(meteo, delimiter='\t')
        meteo["time"] = times.str2datetime(meteo["time"], "%Y-%m-%dT%H:%M:%S")
        alt_text = meteo
    else:
        alt_text = 'vg_sim_data'
    meteo_elcom = open(met, 'w')
    date = times.datetime2unix(meteo["time"])
    sw = np.array(meteo["Qsw"], dtype=float)
    sw[np.where(sw < 0)] = 0
    lw = np.array(meteo["ILWR"], dtype=float)
    at = np.array(meteo["theta"], dtype=float)
    try:
        e = np.array(meteo["e"], dtype=float)
        rh = mxy.vap_p2rel(e, at)
    except KeyError:
        rh = np.array(meteo["rh"], dtype=float)
    kopf(meteo_elcom, 'monty.vg2elcom', alt_text, info=info,
        varis=['TIME', 'AIR_TEMP', 'REL_HUM', 'SOLAR_RAD', 'LW_RAD_IN'])
    date = times.unix2cwr(date)
    rh = np.where(rh > 1, 1, rh)
    data = np.array((date, at, rh, sw, lw)).transpose(1, 0)
    np.savetxt(meteo_elcom, data, fmt='%10.2f\t%5.1f\t%4.2f\t%7.2f\t%6.1f')
    meteo_elcom.close()
    if wind:
        wind = open(wind, 'w')
        try:
            u = np.array(meteo["U"], dtype=float)
            u[np.where(u < 0)] = 0
            if ts != 86400:
                u = u.repeat(old_div(86400, ts))
            u = windfaktor * u
            data = np.array((date, u)).transpose(1, 0)
            np.savetxt(wind, data, fmt='%10.2f\t%5.2f')
        except KeyError:
            u = np.array(meteo["u"], dtype=float)
            v = np.array(meteo["v"], dtype=float)
            print('Wind in Komponenten')
            dirn, speed = avrwind.component2angle(u, v, wind=True)
            speed = windfaktor * speed
            data = np.array((date, speed, dirn)).transpose(1, 0)
            kopf(wind, 'monty.vg2elcom', alt_text,
                varis=['TIME', 'WIND_SPEED', 'WIND_DIR'],
                info='Wind in Komponenten generiert')
            np.savetxt(wind, data, fmt='%10.2f\t%5.2f\t%5.1f')
        wind.close()


def kopf(neufile, funct, infile, info=None, varis=None, bc_n=0):
    zeit = time.strftime("%d. %B %Y %H:%M", time.localtime())
    print(zeit)
    neufile.write('!--- %s ----------------------------------------\n' % zeit)
    neufile.write('! aus %s erstellt von %s \n' % (infile.split('\\')[-1],
                                                   funct))
    if info == None:
        neufile.write('!---------------------------------------\n')
    else:
        neufile.write('! %s\n' % info)
    nn = len(varis)
    neufile.write('%i data sets\n0 seconds between data\n' % (nn - 1))
    for _ in range(nn):
        neufile.write('      %i' % bc_n)
    neufile.write('\n')
    for var in varis:
        neufile.write('%s\t' % var)
    neufile.write('\n')


def plot_elcom_meteo(meteofile, windfile=None, ma=None, simdict=True):
    if simdict == True:
        date, at = times.datetime2unix(meteofile["time"]), meteofile["theta"]
        rh, sw, lw = meteofile["rh"], meteofile["Qsw"], meteofile["ILWR"]
    else:
        meteo = np.loadtxt(meteofile, skiprows=7,
                           converters={0: times.cwr2unix})
        date, at, rh, sw, lw = \
            meteo[:, 0], meteo[:, 1], meteo[:, 2], meteo[:, 3], meteo[:, 4]
    deltat = date[5] - date[4]
    print('swmax ...')
    date_sw = np.arange(date[1] - deltat, date[-2] + deltat + 1, 3600)
    swdate = times.unix2datetime(date_sw)[:-1]
    swmax = mxy.pot_s_rad(swdate)
    date = times.unix2datetime(date)
    text = ('Temperatur:\ntime, min: %s, %6.2f\n'
            % (date[np.where(at == at.min())][0], at.min()) +
            'time, max: %s, %6.2f\n'
            % (date[np.where(at == at.max())][0], at.max()) +
            'average: %6.2f\nstabw: %6.2f\n'
            % (at.mean(), at.std()) +
            'rel. Feuchte:\naverage: %6.2f\nstabw: %6.2f\n'
            % (rh.mean(), rh.std()) +
            'Solarstrahlung:\ntime, max: %s, %6.2f\n'
            % (date[sw == sw.max()][0], sw.max()) +
            'average: %6.2f\nstabw: %6.2f\n'
            % (sw.mean(), sw.std()) +
            'langwellige Strahlung:\ntime, min: %s, %6.2f\n'
            % (date[lw == lw.min()][0], lw.min()) +
            'time, max: %s, %6.2f\naverage: %6.2f\nstabw: %6.2f\n'
            % (date[lw == lw.max()][0], lw.max(), lw.mean(), lw.std()))
    print(text)

    if windfile or simdict == True:
        if simdict == True:
            dr, sp = avrwind.component2angle(meteofile["u"].astype(np.float64),
                        meteofile["v"].astype(np.float64))  # asstype!
            datew = date
            wtitel = 'VG wind'
        else:
            wind = open(windfile, "r")
            wind_lines = wind.readlines(0)
            wind.close()
            wtitel = str(wind_lines[1])
            line = wind_lines[6]  # header mit variablen
            vars_ = line.strip().split('\t')
            if len(vars_) == 1:
                vars_ = line.strip().split()
            ii_var = {}
            for i, variable in enumerate(vars):
                ii_var[variable] = i
            iis = ii_var['WIND_SPEED']
            iid = ii_var['WIND_DIR']
            wtitel = wtitel[1:].strip()
            werte = np.loadtxt(windfile, skiprows=7, usecols=(0, iis, iid),
                    converters={0: times.cwr2unix})
            datew = times.unix2datetime(werte[:, 0])
            sp, dr = np.array(werte[:, 1]), np.array(werte[:, 2])
        plot_wind(sp, dr)
        text = ('Wind Speed\ntime, max: %s, %6.2f\n'
                % (datew[np.where(sp == np.max(sp))][0], np.max(sp)) +
                'average: %6.2f\nstabw: %6.2f\n'
                % (np.average(sp), (np.var(sp)) ** 0.5)
                + 'Wind Direction\naverage: %5.1f' % np.average(dr))
        print(text)

        fig = plt.figure(figsize=(10, 12))
        nfig = 5
    else:
        fig = plt.figure(figsize=(10, 10))
        nfig = 4
    fig.canvas.set_window_title('meteo')
    plt.subplots_adjust(bottom=0.00, top=0.92, right=0.95, wspace=None,
                        hspace=0.4)
    plt1 = plt.subplot(nfig, 1, 1)  # ### 1
    plt.plot(date, at, 'k', label='Temperature')
    if ma:
        plt.plot(date, smooth(at, window_len=ma, window_function='flat'), 'r',
                 linewidth=2)
    plt.axhline(y=0.0, color='b')
    if simdict:
        plt.setp(plt1, ylabel='$^{\circ}$C', title='VG meteo\nTemperature')
    else:
        plt.setp(plt1, ylabel='$^{\circ}$C',
                 title='%s\nTemperature' % meteofile)
    plt.grid('on')
    plt2 = plt.subplot(nfig, 1, 2, sharex=plt1)  # ### 2
    plt.plot(date, rh, 'c', label='rel. humidity')
    plt.grid('on')
    plt.setp(plt2, ylim=(0, 1), title='\nrel. humidity')
    plt2 = plt.subplot(nfig, 1, 3, sharex=plt1)  # ### 3
    plt.plot(date, sw, 'b', label='sw radiation')
    plt.plot(swdate, swmax, c=[0.8, 0.8, 0.8], alpha=0.5)
    plt.grid('on')
    plt.setp(plt2, ylabel='W/m$^{2}$', title='\nsw radiation')
    plt2 = plt.subplot(nfig, 1, 4, sharex=plt1)  # ### 4
    plt.plot(date, lw, 'r', label='lw radiation')
    if ma:
        plt.plot(date, smooth(lw, window_len=ma, window_function='flat'), 'k',
                 linewidth=2)
    plt.grid('on')
    plt.setp(plt2, ylim=(100, 500), ylabel='W/m$^{2}$', title='\nlw radiation')
    plt.matplotlib.dates.AutoDateLocator()
    plt.gcf().autofmt_xdate(rotation=45)
    if windfile or simdict == True:
        ax1 = plt.subplot(nfig, 1, 5, sharex=plt1)  # ### 5
        plt.setp(ax1, ylim=(0, 360), title='\n' + wtitel)
        ax1.plot(datew, dr, color=[0.5, 0.5, 0.5], label='wind direction')
        plt.grid('on')
        plt.matplotlib.dates.AutoDateLocator()
        plt.gcf().autofmt_xdate(rotation=45)
        ax2 = ax1.twinx()
        ax2.plot(datew, sp, 'g', label='wind speed')
        ax2.set_ylabel('m/s')


def plot_wind(sp, dr):
    plt.figure()
    _ = plt.hist(sp, bins=41, range=(0, 24), density=True)
    plt.title('wind speed vg generated wind')
    plt.grid()
    windrose.windrose(dr, n_sectors=36)
    plt.title('wind direction vg generated wind')


def keep_random_state(pth, c_s_list,
                      varnames=("Qsw", "ILWR", "theta", "rh", "u", "v"),
                      varnames_dis=("Qsw", "u", "v"), p=3, **kwargs):
    """ produces a set (len(c_s_list)) of meteorological data files (ELCOM bc)
    with same randomness and user-defined disturbances

    Parameters
    ----------
    pth : string
        path to folder to store output files
    c_s_list : list of np.arrays
        user-defined disturbances of air temperature
    varnames : tuple of str, optional
        variables to simulate
    varnames_dis : tuple of str, optional
        variables to disaggregate to hourly values
    p : int, optional
        Autoregressive order of the VAR-process. (number of previous days to
        consider).
        3 was suitable for a DWD-Constance dataset.
    """
    met_vg = vg.VG(varnames, plot=False)
    met_vg.fit(p=p)
    for i_, c_s in enumerate(c_s_list):
        if i_ == 0:
            random_state = None
        else:
            p = met_vg.p
            if met_vg.q == None:
                q = 0
            else:
                q = met_vg.q
            random_state = os.path.join(vg.conf.data_dir,
                                        'VARMA_p%i_q%i_sim.random_state'
                                        % (p, q))
            print(random_state)
        metfile, windfile = 'meteo_vg_%03d.d' % i_, 'input_wind_vg_%03d.d' % i_
        times_, sim = met_vg.simulate(climate_signal=c_s,
                                      random_state=random_state, **kwargs)
        # simulate sometimes produces NANs. We don't want them. So if there are
        # any, try again:
        while ~np.isfinite(np.average(sim[2, :])):
            times_, sim = met_vg.simulate(climate_signal=c_s,
                                          random_state=random_state, **kwargs)
        if varnames_dis:
            times_, sim = met_vg.disaggregate(varnames_dis)
        # make dictionary:
        sim = np.append(sim, times_).reshape(len(varnames) + 1, -1)
        varnames_ = varnames + ("time",)
        sim_ = dict((varn.strip(), simi) for varn, simi in zip(varnames_, sim))
        met = os.path.join(pth, metfile)
        wind = os.path.join(pth, windfile)
        vg2elcom(sim_, met, teiler=1.0, ts=3600, wind=wind, windfaktor=1.3,
                    simdict=True, info=str(kwargs))
    # save random_state:
    random_save = os.path.join(pth, '%s_p%i_q%i.random_state'
                               % (metfile[:-5], p, q))
    shutil.copy(random_state, random_save)


def get_means(met_vg=None, varnames=("Qsw", "ILWR", "theta")):
    """ get trigonometric function describing average air temperature from vg
    e.g. to be used as input for make_c_s_list
    """
    try:
        met_vg.data_raw.shape
    except AttributeError:
        # falls Objekt noch nicht existiert
        met_vg = vg.VG(varnames, plot=False)
    try:
        met_vg.sim_doys.shape
    except AttributeError:
        met_vg.fit(p=3)
        _ = met_vg.simulate()
    _T = (2 * np.pi / 365 * met_vg.sim_doys)[np.newaxis, :]
    dist, solution = met_vg.dist_sol[met_vg.primary_var]
    means = old_div(dist.trig2pars(solution, _T)[0], 24.)
    return means


def make_c_s_list(means, winter=None, faktoren=None, jahr=2):
    """ produce a list of disturbed mean air temperatures ('climate_signals')
    e.g. to be used as input for keep_random_state

    'Winter' will be added to the 3rd (if jahr==2), winter: jd 300 - jd 95

    Parameters
    ----------
    means : np.array
        undisturbed mean air temperatures, e.g. from get_means
    winter : np.array of shape (160,)
        deviance from means for 160 days
    faktoren : list of ints or None
        will be replaced with range(5) if None.
    jahr : float
        which part of time series to change

    Returns
    -------
    c_s_list : list of np.arrays
        set of climate signals as disturbance for vg
        """
    if faktoren is None:
        faktoren = list(range(5))
    if winter is None:
        winter = np.append(-np.arange(0, 1, 0.0125), np.arange(-1, 0, 0.0125))
    c_s_list = []
    for faktor in faktoren:
        c_s = np.copy(means)
        c_s[366 + 365 * (jahr - 2) + 300:366 + 365 * (jahr - 2) + 460] += \
            winter * faktor
        c_s_list.append(c_s)
    return c_s_list


if __name__ == "__main__":
    import config_kinneret
    vg.conf = config_kinneret
#     vg.delete_cache()
    np.random.seed(0)
    met_vg = vg.VG(("theta", "Qsw", "rh", "u", "v"), verbose=True)
    met_vg.fit(3)
    met_vg.simulate()
    met_vg.disaggregate()
    met_vg.plot_daily_cycles()
    plt.show()
    met_vg.to_glm("/home/dirk/data/Kinneret/glm/met_vg_hourly4.csv")
    #main(2, "/tmp", T=35064 / 24 + 24)
