How-To
######

.. Contents::

This tutorial shows how to use VG in an ipython shell. The data used for the examples is provided in the ``sample.met`` file. Everything here should work without any changes to the ``config_template.py`` file.

Another tutorial shows how to configure VG to use other data.

Initialization
**************

.. ipython:: :okwarning:

    In [1]: import vg
    
    In [2]: my_vg = vg.VG(("theta", "Qsw", "ILWR", "rh", "u", "v"), verbose=False)

Initialization involves converting the input to stationary, standard- normal distributed variables via doy-specific quantile-quantile transform. As this is time-consuming, the fitting results are cached on the hard-drive (in the ``conf.cache_dir`` to be exact).

More information on all the initialization parameters: :func:`vg.VG.__init__`.

The :class:`vg.VG` class provides a number of methods for plotting various results. To see the input time series one can use :func:`vg.VG.plot_timehists`:

.. ipython:: :okwarning:

    # i am using the optional figsize parameter here simply to make the figures
    # more beautiful
    @savefig meteogram.png
    In [3]: figs, axes = my_vg.plot_meteogram_daily(figsize=(8, 6))

The fitting of the daily seasonal distribution can be visualized with :func:`vg.VG.plot_daily_fit`:

.. ipython:: :okwarning:

    # the plot_quantiles and plot_fourier parameters can be omitted in order to
    # get more information on the fitting.
    @savefig seasonal_fit_theta.png
    In [3]: my_vg.plot_daily_fit("theta", plot_quantiles=False, plot_fourier=False)
     
    # if the fit is good, the histogram of the quantiles should be roughly
    # uniformly distributed
    @savefig seasonal_fit_theta_quantiles.png
    In [4]: my_vg.plot_daily_fit("theta", plot_fourier=False)

If you are not satisfied with the conversion, adjust the settings in ``config.py`` and initialize VG again with the ``refit``-parameter:

.. ipython:: :okwarning:

    In [3]: my_vg = vg.VG(("theta", "Qsw", "ILWR", "rh", "u", "v"), refit="ILWR", verbose=False)
    
    # you can also supply multiple variables to refit
    In [4]: my_vg = vg.VG(("theta", "Qsw", "ILWR", "rh", "u", "v"), refit=("ILWR", "rh"), verbose=False) 

Fitting
*******

Calling :func:`vg.VG.fit` fits the stochastic process to the transformed data. When called without parameters, an order selection is performed to find a good compromise between the fit and the number of parameters. Per default, the moving average part is neglected (``q=0``).

.. ipython:: :okwarning:

    In [3]: my_vg.fit()

    # the autoregressive order can be fixed like this
    In [7]: my_vg.fit(3)

Simulation
**********

Without parameters, :func:`vg.VG.simulate` generates time series similar to the input data.

.. ipython:: :okwarning:

    In [8]: times_out, sim_data = my_vg.simulate()

The output is also stored in the ``out_dir`` (specified in ``config.py``) as text file.

At this point it can be assessed whether the order selection was successful. :func:`vg.VG.plot_autocorr` provides a shortcut to plot the autocorrelations of residuals, measured (continuous line) and simulated (dashed line) data (in the "real" and the "transformed" domain)

.. ipython:: :okwarning:

    # in a real ipython shell one call to plot_autocorr suffices. here i have
    # to hack to get all figures
    @savefig autocorr_stale_1.png
    In [9]: figs = my_vg.plot_autocorr()

    @savefig autocorr_stale_2.png
    In [9]: vg.plt.figure(5)
    
Would the fit have been less good, one could consider calling :func:`vg.VG.fit` again with a higher ``p``.

Scenarios
=========

Scenarios are implemented through changes based on the primary variable (default: air temperature). The primary variable can be specified by the parameter ``primary_var`` in :func:`vg.VG.simulate`.

Increased mean (``theta_incr``)
-------------------------------

.. ipython:: :okwarning:
    
    In [9]: times_out, sim_data = my_vg.simulate(theta_incr=4)
    
    # we can display the result like we did above with the input data
    @savefig meteogram_sim_theta.png
    In [10]: figs, axes = my_vg.plot_meteogram_daily()

Another way to visualize the simulation is offered by the method :func:`vg.VG.plot_doy_scatter`:

.. ipython:: :okwarning:

    @savefig doy_scatter_theta.png
    In [12]: my_vg.plot_doy_scatter("theta", figsize=(8, 4))


Increased variability via enhanced episodes (``mean_arrival`` and ``disturbance_std``)
--------------------------------------------------------------------------------------

For increased variability, a Poisson-process is used to set the theoretical mean of the autoregressive process. Durations of episodes are drawn from an exponential distribution with the mean specified as ``mean_arrival``. For each episode, a disturbance is drawn from a normal distribution with the standard deviation of ``disturbance_std``.

.. ipython:: :okwarning:
    
    In [11]: times_out, sim_data = my_vg.simulate(mean_arrival=7, disturbance_std=4)


Providing a climate signal (``climate_signal``)
-----------------------------------------------

VG can be made to follow a specific signal by passing an array with the ``climate_signal`` parameter.

Output
======

:func:`vg.VG.simulate` dumps the generated time series in the ``out_dir`` (specified in ``config.py``) as an ascii file. The filename is chosen by the fitting parameters (Example: ``VARMA_p3_q0_sim.dat``)

Disaggregation
**************

See :func:`vg.VG.disaggregate`

.. ipython:: :okwarning:

    In [12]: times_dis, sim_dis = my_vg.disaggregate(("Qsw", "u", "v"))

Disaggregation also regenerates the seasonal changes in daily cycle.

.. ipython:: :okwarning:

    @savefig daily_cycles.png
    In [13]: fig, axes = my_vg.plot_daily_cycles("Qsw")

Tips
****

- Caching: Once you change anything in the met_file you have to empty the cache! To empty the cache, simply delete everything in the ``cache_dir`` (specified in ``config.py``) or call the function :func:`vg.delete_cache`
