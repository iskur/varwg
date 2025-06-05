"""Central configuration file for all things VG.
This dummy-file contains necessary variables used as configurations and
suggestions for their values in the form of comments

Adjust this file for your needs and rename it to "config.py".
"""

import os
import tempfile

# -----------------------------------------------------------------------------#
# coordinates of measurement station
# they are used to determine the maximum possible short-wave radiation
# the following values point to constance, germany
latitude = 47.66
longitude = 9.18
# -----------------------------------------------------------------------------#
# by defining this variable we can set other directories relative to it (see
# data_dir)
script_home = os.path.dirname(__file__)
# data_dir = os.path.join(script_home, "MyProject", "data")
# data_dir = "i am just a dummy"
data_dir = os.path.join(tempfile.gettempdir(), "VG")

# the met_file can also be specified when initializing the VG object
# i.e. 'my_vg = vg.VG(var_names, met_file="mymetfile")'
# met_file = "i am just a dummy"
met_file = os.path.join(script_home, "sample.met")
# -----------------------------------------------------------------------------#
cache_dir = os.path.join(data_dir, "cache")
out_dir = data_dir
seasonal_cache_file = os.path.join(cache_dir, "seasonal_solutions.she")
# this is a very experimental feature
# hyd_file = os.path.join(data_dir, "rhein_und_br_a_interp_jd.txt")
# -----------------------------------------------------------------------------#
# further suggestion: uncomment this to ensure the existence of the needed
# directories
# for directory in (data_dir, out_dir, cache_dir):
#    if (directory != "i am just a dummy") and (not os.path.exists(directory)):
#        os.makedirs(directory)


# -----------------------------------------------------------------------------#
# Configuration of how we call things
# -----------------------------------------------------------------------------#

import collections

long_var_names = {
    "R": "Precipitation",
    "U": "Wind velocity",
    "u": "Eastward wind speed",
    "v": "Northward wind speed",
    "ILWR": "Incident long-wave radiation",
    "Qsw": "Short-wave radiation",
    "Qsw_lars": "Short-wave radiation",
    "Cloud_Cover": "Cloud cover",
    "theta": "Temperature",
    "theta_lars": "Temperature",
    "theta_monthly": "Monthly averaged Temperature",
    "e": "Vapour pressure",
    "rh": "Relative humidity",
    "nao": "North Atlantic Oscillation Index",
    "wtemp_bregenzerach": "Water temperature of Bregenzer Ach",
    "wtemp_rhein": "Water temperature of Rhine",
}
units = collections.defaultdict(lambda: r"-")
units.update(
    {
        "ILWR": "[$W / m ^ 2$]",
        "e": "[hPa]",
        "rh": "[-]",
        "Cloud_Cover": "[-]",
        "R": "[mm]",
        "Qsw": "[$W / m ^ 2$]",
        "Qsw_lars": "[$W / m ^ 2$]",
        "theta": "[$^{\circ}C$]",
        "theta_lars": "[$^{\circ}C$]",
        "theta_monthly": "[$^{\circ}C$]",
        "U": "[$m / s$]",
        "u": "[$m / s$]",
        "v": "[$m / s$]",
        "nao": "?",
    }
)
# long variable names together with units
ylabels = [
    "%s %s" % (long_var_names[var_name], units[var_name])
    for var_name in long_var_names
]
ygreek = collections.defaultdict(lambda: r"-")
ygreek.update(
    {
        "R": r"$R$",
        "theta": r"$\theta$",
        "rh": r"$\phi$",
        "Qsw": r"$Q_{sw}$",
        "ILWR": r"$Q_{lw(inc.)}$",
        "u": r"$u$",
        "v": r"$v$",
    }
)


def var_names_greek(var_names):
    return [ygreek[var_name] for var_name in var_names]


# -----------------------------------------------------------------------------#
# These are internals of the fitting / deseasonalization process
# -----------------------------------------------------------------------------#

import numpy as np
from vg.time_series_analysis import distributions
from vg.time_series_analysis import seasonal_distributions as sd
from vg.time_series_analysis import seasonal_kde as skde
from vg.meteo import meteox2y
from vg import helpers as my

# theoretical distributions for each variable.
dists = {
    "ILWR": distributions.norm,
    "e": distributions.norm,
    "rh": distributions.truncnorm,
    "Cloud_Cover": distributions.truncnorm,
    "Qsw": "empirical",
    "theta": distributions.norm,
    "R": (distributions.RainMix, distributions.kumaraswamy),
    "U": distributions.lognormal,
    "u": "empirical",
    "v": "empirical",
    "nao": distributions.norm,
}
seasonal_classes = dict.fromkeys(dists, sd.SlidingDist)
seasonal_classes["Qsw"] = skde.SeasonalKDE
seasonal_classes["u"] = seasonal_classes["v"] = skde.SeasonalKDE

# threshold for daily precipitation sum to count as such. same unit as
# precipitation in the input file
threshold = 0.0005
dists_kwds = {
    "R": dict(
        threshold=threshold,
        q_thresh_lower=0.7,
        q_thresh_upper=0.99,
        doy_width=20,
        fft_order=4,
        tabulate_cdf=True,
    ),
    # "R": dict(threshold=1.5e-4,
    #           doy_width=35, fft_order=4),
    "rh": dict(doy_width=10),
}


# this is a helper function for the definition of par_known
# functions that return known parameters to fix those values during fitting
def array_gen(scalar):
    return lambda tt: np.full_like(tt, scalar)


# uncomment the following decorator to cache results of max_qsw on the
# hard-drive and speed up repeated vg initialization
# @my.pickle_cache(os.path.join(cache_dir, "max_qsw_%s.pkl"), warn=False)
def max_qsw(doys, lon=longitude, lat=latitude, **kwds):
    """Use meteo.meteox2y to sum up hourly values of solar radiation for the
    specified days - of - year."""
    doys = np.atleast_1d(doys)
    qsw_2d = -1 * np.ones((22 - 4, len(doys)))
    for i, doy in enumerate(doys):
        date = [str("%03d %02d" % (doy, hour)) for hour in range(4, 22)]
        qsw_2d[:, i] = meteox2y.pot_s_rad(
            date, longt=lon, lat=lat, in_format="%j %H", **kwds
        )
    return qsw_2d.sum(axis=0).ravel()


# read: hours per day
hpd = 24
par_known = dict.fromkeys(dists)
par_known.update(
    {
        "Cloud_Cover": {"l": array_gen(-0.1), "u": array_gen(hpd + 0.1)},
        "Qsw": {
            "l": array_gen(0),
            # the .725 is unfortunately a complicated
            # subject it seems as it should also be a
            # function of t
            "u": np.vectorize(
                lambda x: 0.725 * max_qsw(x + 1 if x < 366 else 1)
            ),
        },
        "rh": {
            "lc": array_gen(-0.001),
            "uc": array_gen(hpd * 1.01),
            # "l": array_gen(-.001),
            # "u": array_gen(hpd * 1.01)
        },
        "R": {
            "l": array_gen(0.0),
            "u": array_gen(1.0),
            # "kernel_bounds": array_gen([0.1, None]),
            # "lc": array_gen(.0002),
            # "uc": array_gen(1e12)
        },
    }
)


# some of the variables have to be treated differently in the hourly
# discretization
seasonal_classes_hourly = {key: val for key, val in seasonal_classes.items()}
# seasonal_classes_hourly["rh"] = skde.SeasonalHourlyKDE
dists_hourly = {key: val for key, val in list(dists.items())}
dists_hourly.update({"rh": distributions.Censored(distributions.norm)})

par_known_hourly = {key: val for key, val in list(par_known.items())}
par_known_hourly.update(
    {
        "rh": {
            "lc": array_gen(-0.001),
            "uc": array_gen(1.01),
        },
        "R": {
            "uc": array_gen(15.0),
            # "lc": array_gen(old_div(dists_kwds["R"]["threshold"], hpd)),
        },
        "Qsw": {
            "lc": array_gen(0),
            # the .725 is unfortunately a complicated subject it seems
            # as it should also be a function of t
            "uc": lambda doy: 0.725 * meteox2y.pot_s_rad(doy),
            # "uc": np.vectorize(lambda x: hpd ** -1 * .725 *
            #                    max_qsw(x + 1 if x < 366
            #                            else 1))
        },
    }
)

# -----------------------------------------------------------------------------#
# Here you can define conversions done after the simulation and before
# output of the data.
# The functions should accept a datetime array an array with the data and
# an iterable containing the variable names.
# -----------------------------------------------------------------------------#

conversions = []


def R_m2mm(times_, data, var_names, inverse=False):
    try:
        ri = var_names.index("R")
    except ValueError:
        return times_, data, var_names
    if inverse:
        data[ri] /= 1000 * 24
    else:
        data[ri] *= 1000 * 24
    return times_, data, var_names


conversions += [R_m2mm]

# # for example: calculate long-wave radiation from air temperature
# def theta2ilwr(times_, data, var_names):
#     from scipy import constants
#     theta = data[var_names.index("theta")]
#     theta_kelvin = constants.C2K(theta)
#     e = 6.2 * np.exp(17.26 * theta / (theta_kelvin - 35.8))
#     boltz = 0.0000000567  # ask gideon
#     alpha, beta = .42, .065
#     ilwr = boltz * theta_kelvin ** 4 * (alpha + beta * np.sqrt(e))
#     if "ILWR" not in var_names:
#         data = np.vstack((data, ilwr))
#         var_names = tuple(list(var_names) + ["ILWR"])
#     else:
#         data[var_names.index("ILWR")] = ilwr
#     return times_, data, var_names
#
# conversions += [theta2ilwr]


# -----------------------------------------------------------------------------#
# Define the format the output should have
# -----------------------------------------------------------------------------#

# if not specified differently, output will have 3 decimal places
out_format = collections.defaultdict(lambda: "%.3f")
out_format.update(
    {
        "theta": "%.3f",
        "Qsw": "%.3f",
        "ILWR": "%.3f",
        "rh": "%.3f",
        "U": "%.3f",
        "wdir": "%.1f",
        "u": "%.3f",
        "v": "%.3f",
        "R": "%.6f",
    }
)
