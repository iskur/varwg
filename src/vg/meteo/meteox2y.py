"""
Meteorological Conversions (:mod:`meteo.meteox2y`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The meteox2y module provides functions to calculate derived meteorological
variables from measured meteorological values.

.. currentmodule:: vg.meteo.meteox2y

.. autosummary::
   :nosignatures:
   :toctree: generated/

   sat_vap_p
   rel2vap_p
   vap_p2rel
   dewpoint
   dew2rel
   norm_pressure
   iziomon
   lw2clouds
   lw_tennessee
   haude
   turc
   turc_rad
   hargreaves
   penman_monteith
   pot_s_rad
   sunshine
   blackbody_rad
   altitude
   spec_hum
   psychro2e
   slope_sat_p

"""

import warnings
from datetime import datetime, timedelta

import bottleneck
import numpy as np
import pytz
import xarray as xr
from scipy import stats
from timezonefinder import TimezoneFinder

import vg
from vg import helpers as my

tzf = TimezoneFinder()


def sat_vap_p(at):
    """saturation vapour pressure from air temperature

    Parameters
    ----------
    at : float or numpy.array of floats
        air temperature [deg C]

    Returns
    -------
    c_e : float or numpy.array of floats
        saturation vapour pressure [hPa]

    References
    ----------
    Hydrologie-Skript I, p. 17

    Examples
    --------
    >>> sat_vap_p(25.0)
    31.688149728170984

    >>> at = np.array((0.0, 2.5, 5.0, 7.5, 10.0))
    >>> sat_vap_p(at[:3])
    array([6.11      ,  7.31533365,  8.72596589])
    >>> sat_vap_p(at[3:])
    array([10.3711941 ,  12.28364703])
    """
    at = np.array(at)
    e0 = 6.11
    a = 17.27
    b = 237.3
    c_e = e0 * np.exp(a * at / (b + at))
    return c_e


def rel2vap_p(rh, at):
    """vapour pressure from relative humidity and air temperature

    Parameters
    ----------
    rh : float or numpy.array of floats
        relative humidity with values between 0 and 1 [-]
    at : float or numpy.array of floats
        air temperature [deg C]

    Returns
    -------
    e : float or numpy.array of floats
        vapour pressure [hPa]

    Examples
    --------
    >>> rel2vap_p(0.5,20.0)
    11.695234581996313

    >>> rh = np.array((0.9, 0.8, 0.7, 0.6, 0.5))
    >>> at = np.array((0.0, 2.5, 5.0, 7.5, 10.0))
    >>> rel2vap_p(rh,at)
    array([5.499     ,  5.85226692,  6.10817613,  6.22271646,  6.14182351])

    """
    rh, at = np.array(rh), np.array(at)
    c_e = sat_vap_p(at)
    e = rh * c_e
    return e


def rel2abs_hum(rh, at):
    """Absolute humidity from relative humidity and air temperature.

    Parameters
    ----------
    rh : float or numpy.array of floats
        relative humidity with values between 0 and 1
    at : float or numpy.array of floats
        air temperature [deg C]

    Returns
    -------
    abs_hum : float or numpy.array of floats
        absolute humidity [g / m^3]

    Examples
    --------
    >>> rel2abs_hum(0.8, 20)
    13.831072059995932
    """
    rh, at = map(np.array, (rh, at))
    e = rel2vap_p(rh, at) * 100
    return 2.16679 * e / (273.15 + at)


def abs_hum2rel(abs_hum, at):
    """Relative humidity from absolute humidity and air temperature.

    Parameters
    ----------
    abs_hum : float or numpy.array of floats
        absolute humidity [g / m^3]
    at : float or numpy.array of floats
        air temperature [deg C]

    Returns
    -------
    rh : float or numpy.array of floats
        relative humidity with values between 0 and 1

    Examples
    --------
    >>> abs_hum2rel(13.831072059995932, 20)
    0.8
    """
    abs_hum, at = map(np.array, (abs_hum, at))
    e = abs_hum / 2.16679 * (273.15 + at)
    return vap_p2rel(e / 100, at)


def vap_p2rel(e, at):
    """relative humidity from vapour pressure and air temperature

    Parameters
    ----------
    e : float or numpy.array of floats
        vapour pressure [hPa]
    at : float or numpy.array of floats
        air temperature [deg C]

    Returns
    -------
    rh : float or numpy.array of floats
        relative humidity with values between 0 and 1 [-]

    Examples
    --------
    >>> vap_p2rel(6.22,20.0)
    0.26592027532201407

    >>> e = np.array((5.5, 5.8, 6.1, 6.2, 6.3))
    >>> at = np.array((0.0, 2.5, 5.0, 7.5, 10.0))
    >>> vap_p2rel(e,at)
    array([0.90016367, 0.79285516, 0.69906301, 0.59780966, 0.512877  ])
    """
    e, at = np.array(e), np.array(at)
    c_e = sat_vap_p(at)
    rh = e / c_e
    return rh


def dewpoint(at, rh=None, e=None):
    """dewpoint from air temperature and humidity

    As input is required: air temperature (`at`) and EITHER relative humidity
    (`rh`) OR vapour pressure (`e`).

    Parameters
    ----------
    at : float or numpy.array of floats
        air temperature [deg C]
    rh : float or numpy.array of floats or None
        relative humidity with values between 0 and 1 [-]
    e : float or numpy.array of floats or None
        vapour pressure [hPa]

    Returns
    -------
    dew : float or numpy.array of floats
        dewpoint [deg C]

    Raises
    ------
    Warning
        If relative humidity is > 1.1 (e. g. if vapour pressure is taken as
        relative humidity)

    Examples
    --------
    >>> dewpoint(20.,rh=0.5)
    9.269628637124908

    >>> dewpoint(20.,e=10.0)
    6.968196840688138

    >>> at = np.array((0.0, 2.5, 5.0, 7.5, 10.0))
    >>> rh = np.array((0.9, 0.8, 0.7, 0.6, 0.5))
    >>> dewpoint(at, rh=rh)
    array([-1.43893707, -0.59071344, -0.0041022 ,  0.25144096,  0.07140267])

    >>> at = np.array((0.0, 2.5, 5.0, 7.5, 10.0))
    >>> e = np.array((5.5, 5.8, 6.1, 6.2, 6.3))
    >>> dewpoint(at, e=e)
    array([-1.43646874, -0.71330622, -0.02250498,  0.20109231,  0.42152362])
    """
    at = np.array(at)
    e0 = 6.11
    a = 17.27
    b = 237.3
    if rh is not None:
        e = rel2vap_p(rh, at)
        summ = np.sum(rh > 1.1)
        if summ > 0:
            warnings.warn(" %i relative humidity values are > 1.1" % summ)
    dew = b * (np.log(e) - np.log(e0)) / (a - (np.log(e) - np.log(e0)))
    return dew


def dew2rel(dew, at):
    """relative humidity from dewpoint and air temperature

    Parameters
    ----------
    at : float or numpy.array of floats
        air temperature [deg C]
    dew : float or numpy.array of floats
        dewpoint [deg C]

    Returns
    -------
    rh : float or numpy.array of floats
        relative humidity with values between 0 and 1 [-]

    Examples
    --------
    >>> dew2rel(15.,16.0)
    0.9378863357566809

    >>> at = np.array((0.0, 2.5, 5.0, 7.5, 10.0))
    >>> dew = np.array((0.0, 3., 5.0, 7.7, 10.0))
    >>> dew2rel(at,dew)
    array([1.        ,  0.96506519,  1.        ,  0.98642692,  1.        ])
    """
    dew, at = np.array(dew), np.array(at)
    e = sat_vap_p(dew)
    c_e = sat_vap_p(at)
    rh = e / c_e
    return rh


def norm_pressure(p, at, h=454.0):
    """normalize pressure to sealevel

    formula from wikipedia.de for linear temperature gradient 0.0065K/m

    Parameters
    ----------
    p : float or numpy.array of floats
        pressure [hPa]
    at : float or numpy.array of floats
        air temperature [deg C]
    h : float, optional
        height above sea level of measuring station [m]
        default value = 454.0 (height of station Stuttgart-Lauchaecker)

    Returns
    -------
    p_nn : float or numpy.array of floats
        sea level pressure [hPa]

    References
    ----------

    Examples
    --------
    >>> "%.9f" % norm_pressure(930,10.0)
    '982.074383073'

    >>> "%.9f" % norm_pressure(930,10.0,h=765)
    '1019.090035371'

    >>> at = np.array((0.0, 2.5, 5.0, 7.5, 10.0))
    >>> p = np.array((925, 930, 935, 932, 931))
    >>> norm_pressure(p,at,h=765)
    array([1016.98077228,  1021.60706988,  1026.24032953,  1022.10690872,
            1020.18583111])
    """
    tk = at + 273.16  # T in Kelvin
    p_nn = p * (1 + 0.0065 * h / tk) ** 5.255
    return p_nn


def iziomon(temp, clouds, rh=None, dew=None, e=None, site="low"):
    """incident long wave radiation following Iziomon et al (2003)[1]_

    As input is required: air temperature (`temp`), cloud cover (`clouds`) and
    EITHER relative humidity (`rh`) OR dewpoint (`dew`) OR vapour pressure
    (`e`).

    Parameters
    ----------
    temp : float or numpy.array of floats
        air temperature [deg C]
    clouds : float or numpy.array of floats
        cloud cover with values between 0 and 1 [-]
    rh : float or numpy.array of floats or None
        relative humidity with values between 0 and 1 [-]
    dew : float or numpy.array of floats or None
        dewpoint [deg C]
    e : float or numpy.array of floats or None
        vapour pressure [hPa]
    site : {'low', 'high'}
        parameterisation for lowland or highland site

    Returns
    -------
    lw : float or numpy.array of floats
        incident longwave radiation [W/m**2]

    See Also
    --------
    lw2clouds : reverse (get cloud cover out of long wave, temperature and
        humidity)

    Notes
    -----
    Empirical formula, found for experiments in Bremgarten (47deg54'35''N;
    7deg37'18''E) in the Upper Rhine plain in Germany (lowland site) and
    Feldberg, 1489 m asl, 47deg52'31''N, 8deg00'11''E, Black Forest, Germany
    (highland site)

    References
    ----------
    .. [1] Iziomon, M.G., Mayer H, Matzarakis A. (2003): Downward atmospheric
       longwave irradiance under clear and cloudy skies: Measurement and
       parameterization, Journal of Atmospheric and Solar-Terrestrial Physics
       65 (2003) 1107 - 1116

    Examples
    --------
    >>> iziomon(15.,0.5,rh=0.89)
    327.52426791875763

    >>> temp = np.array((0.0, 2.5, 5.0, 7.5, 10.0))
    >>> clouds = np.array((0.0, 0.1, 0.8, 0.5, 1.0))
    >>> e = np.array((5.5, 5.8, 6.1, 6.2, 6.3))
    >>> iziomon(temp,clouds,e=e)
    array([225.34414848,  235.0777488 ,  279.01405432,  267.25348612,
            321.15448959])
    """
    # iziomon-parameter lowland site:
    if site == "low":
        Xs, Ys, Zs = 0.35, 10.0, 0.0035
    # mountain site:
    elif site == "high":
        Xs, Ys, Zs = 0.43, 11.5, 0.005
    # temp in Kelvin:
    tk = temp + 273.16
    # humidity in vapour pressure:
    if rh is not None:
        e = rel2vap_p(rh, temp)
    elif dew is not None:
        e = sat_vap_p(dew)
    # clouds in Okta:
    clouds = clouds * 8
    # iziomon:
    lw_clear = blackbody_rad(temp=temp) * (1 - Xs * np.exp(-Ys * e / tk))
    lw = lw_clear * (1 + Zs * clouds**2)
    return lw


def temp2lw(temp):
    """Incident long-wave radiation from air temperature (Gal pc).

    Parameters
    ----------
    temp : float or numpy.array of floats
        air temperature [deg C]

    Returns
    -------
    lw : float or numpy.array of floats
        incident longwave radiation [W/m**2]
    """
    from scipy import constants

    try:
        theta_kelvin = constants.C2K(temp)
    except AttributeError:
        theta_kelvin = constants.convert_temperature(temp, "Celsius", "Kelvin")
    e = 6.2 * np.exp(17.26 * temp / (theta_kelvin - 35.8))
    boltz = 0.0000000567  # ask gideon
    alpha, beta = 0.42, 0.065
    return boltz * theta_kelvin**4 * (alpha + beta * np.sqrt(e))


# dirk: using np.atleast_1d and as scalar here, to make indexing on scalar
# input and the doctests work
@my.asscalar
def lw2clouds(lw, temp, rh=None, dew=None, e=None, site="low"):
    """Cloud cover from incident long wave radiation (Iziomon et al (2003)[1]_)

    As input is required: incident long wave radiation (`lw`),
    air temperature (`temp`) and EITHER relative humidity (`rh`)
    OR dewpoint (`dew`) OR vapour pressure (`e`)

    Parameters
    ----------
    lw : float or numpy.array of floats
        incident longwave radiation [W/m**2]
    temp : float or numpy.array of floats
        air temperature [deg C]
    rh : float or numpy.array of floats or None
        relative humidity with values between 0 and 1 [-]
    dew : float or numpy.array of floats or None
        dewpoint [deg C]
    e : float or numpy.array of floats or None
        vapour pressure [hPa]
    site : {'low', 'high'}
        parameterisation for lowland or highland site

    Returns
    -------
    clouds : float or numpy.array of floats
        cloud cover with values between 0 and 1 [-]

    See Also
    --------
    iziomon : incident long wave radiation from air temperature, cloud cover
        and humidity

    Notes
    -----
    Resulting cloud cover is always between 0 and 1. Gives 0 for unrealistic
    low and 1 for unrealistic high values of `lw`.

    References
    ----------
    .. [1] Iziomon, M.G., Mayer H, Matzarakis A. (2003): Downward atmospheric
       longwave irradiance under clear and cloudy skies: Measurement and
       parameterization, Journal of Atmospheric and Solar-Terrestrial Physics
       65 (2003) 1107 - 1116

    Examples
    --------
    >>> lw2clouds(328., 15., rh=0.9)
    0.49959967427213553

    >>> lw = np.array((225.5, 235.1, 279, 267, 321.2))
    >>> temp = np.array((0.0, 2.5, 5.0, 7.5, 10.0))
    >>> e = np.array((5.5, 5.8, 6.1, 6.2, 6.5))
    >>> lw2clouds(lw,temp,e=e)
    array([0.0555659 ,  0.1020956 ,  0.79983929,  0.49550839,  0.99289638])
    """
    lw, temp = np.atleast_1d(lw, temp)
    # iziomon-parameter lowland site:
    if site == "low":
        Xs, Ys, Zs = 0.35, 10.0, 0.0035
    # mountain site:
    elif site == "high":
        Xs, Ys, Zs = 0.43, 11.5, 0.005
    # temp in Kelvin:
    tk = temp + 273.16
    # humidity in vapour pressure:
    if rh is not None:
        e = np.atleast_1d(rel2vap_p(rh, temp))
    elif dew is not None:
        e = np.atleast_1d(sat_vap_p(dew))
    elif e is not None:
        e = np.atleast_1d(e)
    # iziomon:
    lw_clear = blackbody_rad(temp=temp) * (1 - Xs * np.exp(-Ys * e / tk))
    lw_valid = lw > lw_clear
    n = np.zeros_like(lw)
    n[lw_valid] = ((lw[lw_valid] / lw_clear[lw_valid] - 1) / Zs) ** 0.5
    clouds = n / 8.0
    clouds[np.isnan(clouds)] = 0
    clouds[clouds > 1] = 1
    return clouds


def lw_tennessee(temp, clouds):
    """incident long wave radiation

    Parameters
    ----------
    temp : float or numpy.array of floats
        air temperature [deg C]
    clouds : float or numpy.array of floats
        cloud cover with values between 0 and 1 [-]

    Returns
    -------
    lw : float or numpy.array of floats
        incident longwave radiation [W/m**2]

    See Also
    --------
    iziomon : other empirical formula, from southwestern germany

    References
    ----------
    .. [1] Tennessee Valley Authority 1972. Heat and mass transfer between a
       water surface and the atmosphere Water Resources Research Laboratory
       Report 14, Report No. 0-6803. (I didn't find it. But it's mentioned in:)
    .. [2] ELCOM Science manual
    """
    # temp in Kelvin:
    tk = temp + 273.16
    c_e = 9.37 * 10**-6  # K**-2
    eps_a = c_e * tk**2
    lw = (1 + 0.17 * clouds**2) * eps_a * blackbody_rad(temp=temp)
    return lw


def haude(svp, vp, mon):
    """calculates the evapotranspiration following Haude (1955) [1]_
    as described in the Hydrologie-I-Skript

    Parameters
    ----------
    svp : float or numpy.array of floats
        saturation vapour pressure [hPa], measured at 2pm
    vp : float or numpy.array of floats
        vapour pressure [hPa], measured at 2pm
    mon : int or numpy.array of ints
        month: 1=january, 12=december

    Returns
    -------
    etp : float or numpy.array of floats
        evapotranspiration [mm] daily values

    See Also
    --------
    turc : potential evapotranspiration following Turc

    Notes
    -----
    The Haude formula is only valid in temperate humid climate

    References
    ----------
    .. [1] Haude, W. (1955): Zur Bestimmung der Verdunstung auf moeglichst
       einfache Weise. - Mitt. Dt. Wetterd. 2 (11), Bad Kissingen (Dt.
       Wetterd.)

    Examples
    --------
    >>> haude(12.28,8.83,5)
    1.0005

    >>> haude(12.28,8.83,12)
    0.759

    >>> svp = np.array([11.87809345,  12.28364703,  12.70132647])
    >>> vp = np.array([8.55222728,  8.84422586,  9.14495506])
    >>> haude(svp,vp,12)
    array([0.73169056,  0.75667266,  0.78240171])

    >>> mon = np.array([2,3,4])
    >>> haude(svp,vp,mon)
    array([0.73169056,  0.75667266,  1.03134771])
    """
    mon = mon - 1
    hfs = (
        0.0022,
        0.0022,
        0.0022,
        0.0029,
        0.0029,
        0.0028,
        0.0026,
        0.0029,
        0.0023,
        0.0022,
        0.0022,
        0.0022,
    )  # monthly Haude factor
    try:
        hf = hfs[mon]
    except TypeError:
        hf = np.array([hfs[mo] for mo in mon])
    svp, vp = np.array(svp) * 100, np.array(vp) * 100  # hPa -> Pa
    etp = hf * (svp - vp)
    return etp


def turc(at, ts, mon):
    """calculates the potential evapotranspiration following Turc [1]_ as
    described in the Hydrologie-I-Skript (p. 55)

    Parameters
    ----------
    at : float or numpy.array of floats
        daily average air temperature [degC]
    ts : float or numpy.array of floats
        number of sunshine hours per day
    mon : int or numpy.array of ints
        month: 1=january, 12=december

    Returns
    -------
    etp : float or numpy.array of floats
        potential evapotranspiration [mm] daily values

    See Also
    --------
    haude : evapotranspiration following Haude

    Notes
    -----
    The Turc formula is only valid for `at` > 0 deg C

    References
    ----------
    .. [1] TURC??

    Examples
    --------
    >>> "%.9f" % turc(10,0,5)
    '1.104000000'

    >>> "%.9f" % turc(10,0,12)
    '0.408000000'
    """
    mon = mon - 1
    # Konstanten C1, C2 fuer etwa 53 deg N in Mitteleuropa
    # warum auch immer
    cs = (
        (1.09, 0.18),
        (1.4, 0.259),
        (1.86, 0.35),
        (2.36, 0.429),
        (2.76, 0.476),
        (3, 0.489),
        (2.93, 0.484),
        (2.58, 0.448),
        (2.1, 0.39),
        (1.57, 0.21),
        (1.19, 0.21),
        (1.02, 0.158),
    )
    try:
        c1, c2 = cs[mon]
    except TypeError:
        c1 = np.array([cs[mo][0] for mo in mon])
        c2 = np.array([cs[mo][1] for mo in mon])
    etp = (c1 + c2 * ts) * at / (at + 15)
    return etp


def turc_rad(at, rh, G):
    """calculates the potential evapotranspiration following Turc

    with the global radiation instead of the empiric factors according to
    Hydrologie und Wasserwirtschaft by Maniak

    Parameters
    ----------
    at : numpy.array of floats
        mean daily air temperature [deg C]
    rh : numpy.array of floats
        relative humidity [%]
    G  : numpy.array of floats
        global radiation [W/m^2]

    Returns
    -------
    etp : numpy.array of floats
        potential evapotranspiration [mm/d]

    Notes
    -----
    - For etp < 0.1 mm/d the evapotranspiration rate is set to 0.1 mm/d.
      [DVWK 1996]
    - According to the energetic limit of 7 mm/d in Germany, ETPmax is set to
      7 mm/d.
    """

    rh = np.array(rh, float)
    C = np.ones_like(rh)
    ii = np.where(rh < 50.0)
    C[ii] = 1.0 + ((50.0 - rh[ii]) / 70.0)

    G = G * 86400.0 / 10000.0  # W/m^2 in J/(cm^2*d)

    etp = 0.0031 * C * (G + 209.0) * (at / (at + 15.0))
    ii = np.where(etp > 7.0)
    etp[ii] = 7.0
    jj = np.where(etp < 0.1)
    etp[jj] = 0.1

    return etp


def hargreaves(tmax, tmin, date, in_format="%Y-%m-%dT%H:%M:%S"):
    """calculates the potential evapotranspiration rate [mm/d] following
    Hargreaves & Samani 1985 according to THE ASCE STANDARDIZED REFERENCE
    EVAPOTRANSPIRATION EQUATION

    Parameters
    ----------
    tmax : np.array of floats
        maximum of the daily air temperature [deg C]
    tmin : np.array of floats
        minimum of the daily air temperature [deg C]
    date : np.array of strings
        date strings in format in_format
    in_format : format string
        default: '%Y-%m-%dT%H:%M:%S'

    Returns
    -------
    etp : numpy.array of floats
        potential evapotranspiration [mm/d]

    Notes
    -----
    - equation needs the extraterrestrial radiation Ra, which depends on:

      - inverse relative distance factor for the earth-sun dr []
      - solar declination delta [rad]
      - sunset hour angle [rad]
        with latitude Lauchaecker lat=48.738 deg => pi/180 * 48.738
        deg = 0.8506 rad

    - conversion of date in day of year (j)
    - ETPmax = 7.0 mm/d due to energetic limit (Germany)

    References
    ----------
    """

    j = []
    for dat in date:
        t = datetime.strptime(dat, in_format)
        doy = float(datetime.strftime(t, format="%j"))  # day of year
        j.append(doy)

    # inverse relative distance factor for the earth-sun []
    dr = 1.0 + 0.033 * np.cos(np.multiply((2 * np.pi / 365.0), j))
    # solar declination [rad]
    delta = 0.409 * np.sin(np.multiply((2 * np.pi / 365.0), j) - 1.39)
    # sunset hour angle [rad]
    omega_s = np.arccos(-np.tan(0.8506) * np.tan(delta))
    # extraterrestrial radiation [MJ m^-2 d^-1]
    Ra = (
        24.0
        / np.pi
        * 4.92
        * dr
        * (
            omega_s * np.sin(0.8506) * np.sin(delta)
            + np.cos(0.8506) * np.cos(delta) * np.sin(omega_s)
        )
    )
    # [MJ m^-2 d^-1]
    etp = (
        0.0023
        * (tmax - tmin) ** (0.5)
        * ((tmax + tmin) / 2.0 + 17.8)
        * Ra
        / 2.45
    )  # 2.45: factor for calculating mm/d

    ii = np.where(etp > 7.0)
    etp[ii] = 7.0

    return etp


def penman_monteith(at, u, Rn, rh):
    """Calculates the reference crop evaporation following the Penman-Monteith
    method and the FAO-56 determinations published in ASCE
    Standardized Reference Evapotranspiration Equation

    Parameters
    ----------
    at : np.array of floats
        mean daily air temperature at 2m-height [deg C]
    u  : np.array of floats
        mean daily wind speed at 2m-height [m/s]
    Rn : np.array of floats
        measured net radiation at the crop surface [MJ m^-2 d^-1]
    rh : np.array of floats
        relative humidity [%]

    Returns
    -------
    eto : np.array of floats
        FAO Penman-Monteith standardized reference crop evapotranspiration for
          short (~=0.12m) surfaces [mm d^-1]

    Notes
    -----
    - Units for the 0.408 coefficient are m^2 mm MJ^-1
    - The FAO-56 Penman-Monteith equation is a grass reference equation that
      was derived from the Penman-Monteith form of the combination equation
      (Monteith 1965, 1981) by fixing h = 0.12 m for clipped grass and by
      assuming measurement heights of z = 2 m (at, rh, u) and using a latent
      heat of vaporization of 2.45 MJ kg-1.
      The result is an equation that defines the reference evapotranspiration
      from a hypothetical grass surface having a fixed height of 0.12 m, bulk
      surface resistance of 70 s m-1, and albedo of 0.23.
    - in relationship to the net radiation, the soil heat flux is very small
      and is fixed with G = 0.1*Rn
    """
    rh = rh / 100.0  # relative humidity in decimal[0,1]
    y = 0.000665 * 101.3 * ((293.0 - 0.0065 * 453.0) / 293.0) ** 5.26

    G = 0.1 * Rn
    S = slope_sat_p(at)
    Cn = 900
    Cd = 0.34
    es = sat_vap_p(at)
    ea = rel2vap_p(rh, at)

    eto = (
        0.408 * S * (Rn - G)
        + y * (Cn / (at + 273.15)) * u * ((es - ea) / 10.0)
    ) / (S + y * (1 + Cd * u))

    ii = np.where(eto > 7.0)
    eto[ii] = 7.0

    return eto


def pot_s_rad(
    date,
    lat=48.738,
    longt=9.099,
    in_format="%Y-%m-%dT%H:%M",
    tz_mer=15.0,
    wog=-1,
):
    """theoretical maximal potential solar radiation outside of atmosphere

    Parameters
    ----------
    date : numpy.array of time strings (format: in_format) or datetime objects
        or floats (doys)
    lat : float, optional
        latitude of station in decimal degrees, default: Stuttgart Lauchaecker
    longt : float, optional
        longitude of station in decimal degrees, default: Stuttgart Lauchaecker
    in_format : time format string, optional
        format of date if date is string, default '%Y-%m-%dT%H:%M'
    tz_mer : int, optional
        central meridian of time zone, default: 15 (CET)
    wog : {-1, 1}, optional
        west of greenwich, 1 if west, -1 if east, default = -1

    Returns
    -------
    smax : numpy.array of floats
        maximal potential solar radiation in W/m^2

    Notes
    -----
    from campbell technical note 18 [1]_, except declination of sun (d):
    formula of Spencer (1971) [2]_

    WARNING: Campbell Scientific recommends the use of a high quality sun
    screen lotion when exposing your skin to solar radiation for large values
    of sunshine hours!

    References
    ----------
    .. [1] Campbell Scientific (2005) technical note 18: CALCULATING SUNSHINE
       HOURS FROM PYRANOMETER / SOLARIMETER DATA
    .. [2] Spencer JW (1971) Fourier series representation of the position of
       the Sun. Search 2: 172.

    Examples
    --------
    >>> date_str = np.array(["2011-09-28T11:27"])
    >>> pot_s_rad(date_str)
    array([855.35624182])
    >>> from vg import times
    >>> dt = times.str2datetime(date_str, "%Y-%m-%dT%H:%M")
    >>> pot_s_rad(dt)
    array([855.35624182])
    >>> pot_s_rad(times.datetime2doy(dt))
    array([855.35624182])
    """
    if tz_mer is None:
        timezone_str = tzf.timezone_at(lat=float(lat), lng=float(longt))
        timezone = pytz.timezone(timezone_str)
        dt = datetime(2020, 6, 1)
        tz_offset = (
            timezone.utcoffset(dt, is_dst=False).total_seconds()
        ) / 3600
        tz_mer = 15 * tz_offset
    s0 = 1373  # Solarkonstante W/m^2
    # once upon a time there was a latitude
    Lc = wog * (tz_mer - longt) / 15.0  # Local correction of time
    lat = lat * np.pi / 180  # in rad
    try:
        # where we used to raise an exception
        doys = vg.times.datetime2doy(date)  # if date is datetime
    except (
        TypeError,
        AttributeError,
        ValueError,
        NotImplementedError,
        IndexError,
    ):
        try:
            # ...or two
            # if date is string
            doys = vg.times.datetime2doy(
                vg.times.str2datetime(date, in_format)
            )
        except (TypeError, IndexError):
            # those were the doys my friend, i thought they never end
            doys = date  # if date is already in doys
    hours = (doys - doys.astype(int)) * 24
    # remember how we typecasted away the hours
    doys = doys.astype(int)
    j1 = doys / 100.0
    j2 = (doys - 180) / 100.0
    Et = np.where(
        doys > 180,
        (
            -0.05039
            - 0.33954 * j2
            + 0.04084 * j2**2
            + 1.8928 * j2**3
            - 1.7619 * j2**4
            + 0.4224 * j2**5
        ),
        (
            -0.04056
            - 0.74503 * j1
            + 0.08823 * j1**2
            + 2.0516 * j1**3
            - 1.8111 * j1**4
            + 0.42832 * j1**5
        ),
    )
    t0 = 12 - Lc - Et
    # we sang and dance forever and a doy (less)
    # la la la la lala lala lala lala
    la = 2 * np.pi / 365 * (doys - 1)
    d = (
        0.006918
        - 0.399912 * np.cos(la)
        + 0.070257 * np.sin(la)
        - 0.006758 * np.cos(2 * la)
        + 0.000907 * np.sin(2 * la)
        - 0.002697 * np.cos(3 * la)
        + 0.00148 * np.sin(3 * la)
    )
    sind = np.sin(d)
    sinphi = np.atleast_1d(
        sind * np.sin(lat)
        + np.cos(d) * np.cos(lat) * np.cos(15 * np.pi / 180.0 * (hours - t0))
    )
    sinphi[sinphi < 0] = 0
    smax = s0 * sinphi
    return smax


def pot_s_rad_daily(
    date, lat=48.738, longt=9.099, in_format="%Y-%m-%d", tz_mer=15.0, wog=-1
):
    """daily average values of --> pot_s_rad"""
    # machen wir mal stundenweise:
    if not isinstance(date[0], datetime):
        date = vg.times.str2datetime(date, in_format)
    date_h = np.array(
        [dt__ + timedelta(hours=i) for dt__ in date for i in range(24)]
    )
    pot_h = pot_s_rad(date_h, lat, longt, tz_mer=tz_mer, wog=wog)
    return np.average(pot_h.reshape(-1, 24), axis=1)


def sunshine_pot(doys, lat=48.738, longt=9.099, tz_mer=15.0, wog=-1):
    """Maximum daily sunshine hours based on evaluating pot_s_rad per minute."""
    mins_per_day = 24 * 60
    # doys = doys.astype(float)
    doys = doys[doys == doys.astype(int)]
    doys_minutes = doys.repeat(mins_per_day).reshape(-1, mins_per_day)
    doys_minutes += (np.arange(mins_per_day) / float(mins_per_day))[None, :]
    doys_minutes = doys_minutes.ravel()
    smax = pot_s_rad(doys_minutes, lat, longt, tz_mer=tz_mer, wog=wog)
    smax = smax.reshape(-1, mins_per_day)
    sun_hours = np.sum(smax > 0, axis=1) / 60.0
    return sun_hours


def sunshine(
    sw,
    date,
    lat=48.738,
    longt=9.099,
    in_format="%Y-%m-%dT%H:%M",
    tz_mer=15.0,
    wog=-1,
):
    """sunshine or not?

    calculates maximum potential solar radiation depending on latitude,
    longitude, and time and compares it to actual solar radiation. Sunshine if
    actual solar radiation > 0.4*maximum potential solar radiation
    accurate enough for normal non-scientific use of sunshine hour data

    Parameters
    ----------
    sw : numpy.array of floats
        solar (short wave) radiation in W/m^2
    date : numpy.array of time strings
        date and time in format in_format
    lat : float, optional
        latitude of station in decimal degrees, default: Stuttgart Lauchaecker
    longt : float, optional
        longitude of station in decimal degrees, default: Stuttgart Lauchaecker
    in_format : time format string, optional
        format of date, default '%Y-%m-%dT%H:%M'
    tz_mer : int, optional
        central meridian of time zone, default: 15 (CET)
    wog : {-1, 1}, optional
        west of greenwich, 1 if west, -1 if east, default = -1

    Returns
    -------
    shining : numpy.array containing 0 and 1
        1: sun is shining in corresponding time step, 0: sun is not shining in
        corresponding time step

    Examples
    --------
    >>> sunshine(np.array([450]),np.array(["2011-09-28T11:27"]))
    array([1])
    >>> sunshine(np.array([200]),np.array(["2011-09-28T11:27"]))
    array([0])
    """
    shining = np.zeros_like(sw)
    smax = pot_s_rad(
        date, lat, longt, in_format=in_format, tz_mer=tz_mer, wog=wog
    )
    shining[np.where(sw > 0.4 * smax)] = 1
    return shining


def sonnenscheindauer(date, sw, del_t=60):
    r"""
    bestimmt Sonnenscheindauer anhand der kurzwelligen Solarstrahlung

    Parameters
    ----------
    date  : string
        datetime.datetime(2040, 1, 1, 0, 0, 0)
    sw : string
    del_t : int, optional
        timestep in minutes: 1-min-data is averaged to this timestep.
        Default is 60

    Returns
    -------
    sunshine_hour
    """

    sunshine_min = sunshine(sw, date)
    # sunshine_sum = np.sum(sunshine_min)
    sunshine_hour = sunshine_min / float(del_t)

    return sunshine_hour


def get_tz_offset(dates, longitude, latitude):
    # from tzwhere import tzwhere
    # import pytz
    # tzw = tzwhere.tzwhere()
    # timezone offset
    # timezone_str = tzw.tzNameAt(latitude, longitude)
    if isinstance(dates, datetime):
        dates = (dates,)
    timezone_str = tzf.timezone_at(lat=float(latitude), lng=float(longitude))
    timezone = pytz.timezone(timezone_str)
    dts = [datetime(date.year, date.month, date.day) for date in dates]
    tz_offset = (
        np.array(
            [timezone.utcoffset(dt, is_dst=True).total_seconds() for dt in dts]
        )
        / 86400
    )
    return np.squeeze(tz_offset)


def sunshine_hours(dates, longitude, latitude, tz_offset=None):
    """Calculates hours from sunrise to sunset.

    Notes
    -----
    see:
    https://en.wikipedia.org/wiki/Sunrise_equation
    https://michelanders.blogspot.com/2010/12/calulating-sunrise-and-sunset-in-python.html

    Examples
    --------
    >>> from datetime import date
    >>> sunshine_hours([date(2018, 6, 1)], 8.848, 48.943)  # mühlacker
    array([15.89961046])
    """
    if tz_offset is None:
        tz_offset = get_tz_offset(dates, longitude, latitude)
    J_rise, J_set = sunshine_riseset(dates, longitude, latitude, tz_offset)
    # return (J_set - J_rise) * 60
    return J_set - J_rise


def max_sunshine_minutes(dates, longitude, latitude, tz_offset=None):
    if tz_offset is None:
        tz_offset = get_tz_offset(dates, longitude, latitude)
    hour_of_day = np.array([date.hour for date in dates])
    sun_rise, sun_set = sunshine_riseset(dates, longitude, latitude, tz_offset)
    minutes = np.zeros_like(dates, dtype=float)
    minutes[(hour_of_day > sun_rise) & (hour_of_day < sun_set)] = 60
    # if sunrise happens in the hour_of_day, we have to make minutes
    # smaller
    dawn_mask = (hour_of_day < sun_rise) & (sun_rise < hour_of_day + 1)
    minutes[dawn_mask] = 60 * (sun_rise[dawn_mask] - hour_of_day[dawn_mask])
    # equivalent for the evening
    twilight_mask = (hour_of_day < sun_set) & (sun_set < hour_of_day + 1)
    minutes[twilight_mask] = 60 * (
        hour_of_day[twilight_mask] + 1 - sun_set[twilight_mask]
    )
    assert np.all(minutes >= 0)
    assert np.all(minutes <= 60)
    return minutes


def sunshine_riseset(dates, longitude, latitude, tz_offset=None):
    """Calculates sunrise and sunset hours."""
    if tz_offset is None:
        tz_offset = get_tz_offset(dates, longitude, latitude)
    jdn = vg.times.date2jdn(dates)

    if tz_offset is None:
        tz_offset = get_tz_offset(dates, longitude, latitude)
    n = jdn - 2451545.0008  # current julian day
    J_star = n - longitude / 360 + tz_offset  # mean solar noon
    M = (357.5291 + 0.98560028 * J_star) % 360  # solar mean anomaly
    M_radians = np.radians(M)
    # equation of the center
    C = (
        1.9148 * np.sin(M_radians)
        + 0.02 * np.sin(2 * M_radians)
        + 0.0003 * np.sin(3 * M_radians)
    )
    # ecliptic longitude
    lambda_radians = np.radians((M + C + 282.9372) % 360)
    # solar transit
    J_transit = (
        2451545.5
        + J_star
        + 0.0053 * np.sin(M_radians)
        - 0.0069 * np.sin(2 * lambda_radians)
    )
    # declination of the sun
    sin_del = np.sin(lambda_radians) * np.sin(np.radians(23.44))
    cos_del = np.cos(np.arcsin(sin_del))
    # hour angle
    latitude_radians = np.radians(latitude)
    cos_omega0 = (
        np.sin(np.radians(-0.83)) - np.sin(latitude_radians) * sin_del
    ) / (np.cos(latitude_radians) * cos_del)
    omega0 = np.degrees(np.arccos(cos_omega0)) / 360
    J_transit = J_transit - J_transit.astype(int)
    J_rise = 24 * (J_transit - omega0)
    J_set = 24 * (J_transit + omega0)
    return J_rise, J_set


def blackbody_rad(rad=None, temp=None, eps=1.0):
    """Stefan-Boltzmann law

    There are two ways to use this funtion:

    1) If input is radiation, then the function calculates the absolute
    temperature of the body emitting the radiation

    2) If input is temperature, then the function calculates the radiation
    emitted by the body of this temperature

    Parameters
    ----------
    rad : float or numpy.array of floats or None
        radiation [W/m**2]
    temp : float or numpy.array of floats or None
        temperature [deg C]
    eps : float or numpy.array of floats, optional
        emissivity of a grey body, values between 0 and 1, default=1

    Returns
    -------
    either:
    temp : float or numpy.array of floats or None
        temperature [deg C]
    or:
    rad : float or numpy.array of floats or None
        radiation [W/m**2]

    Examples
    --------
    >>> "%.9f" % blackbody_rad(rad=350)
    '7.138805085'

    >>> "%.9f" % blackbody_rad(temp=0)
    '315.683203500'
    """
    sigma = 5.67 * 10**-8
    tk0 = 273.16
    if rad is not None:
        return (rad / (eps * sigma)) ** 0.25 - tk0
    elif temp is not None:
        return eps * sigma * (temp + tk0) ** 4


def altitude(temp1, temp0, pres1, pres0, alt0):
    r"""Converts continous meassured vertical pressure and temperature data to
    altitude.
    # 1 means i and 0 means i-1

    Parameters
    ----------
    temp1 : float or array_like
        Temperature value at timestep i.
    temp0 : float or array_like
        Temperature value at timestep i-1.
    pres1 : float or array_like
        Pressure value at timestep i.
    pres0 : float or array_like
        Pressure value at timestep i-1.
    alt0 : float or array_like
        Altitude value at timestep i-1.

    Returns
    -------
    altitude : float or array_like
        Altitude at timestep i

    Notes
    -----
    This is the implemented equation:

    g : Gravity acceleration 9.81 m/s^2
    T_0 : Temperature 273.16 K

    .. math::

        altitude = ((\frac{\ln(pres0)}{pres1})*287*((\frac{(\frac{(temp1+temp0)}{2})+T_0)}/{g}))+alt0

    References
    ----------
    *** -> Ask Felix!!! ***

    Examples
    --------
    >>> alt = np.nan*np.ones(40)
    >>> alt[0] = 600
    >>> for i,element in enumerate(alt):    # doctest: +SKIP
    ...    if str(element) == 'nan':    # doctest: +SKIP
    ...        element = altitude(temp[i], temp[i-1], pres[i], pres[i-1], alt[i-1])    # doctest: +SKIP
    """
    tk0 = 273.16
    g = 9.81

    altitude = (
        (np.log(pres0 / pres1)) * 287 * ((((temp0 + temp1) / 2) + tk0) / g)
    ) + alt0
    # ;print 'altitude: ',self.altitude

    return altitude


def spec_hum(e, p):
    """Calculates the specific humidity

    Parameters
    ----------
    pressure : float
        air pressure in hPa
    e : float
        vapour pressure in hPa


    Returns
    -------
    spec_hum : float

    Notes
    -----
    The formula is:
    s = ((0.623*e)/(p-0.377*e))*1000
    where M_w/M_tL=0.622
    and 0.378 = 1-0.622"""

    spec_hum = (0.622 * e * 1000) / (p - 0.378 * e)

    return spec_hum


def psychro2e(t_dry, t_wet, p=None):
    """Vapour pressure from dry and wet temperature from Assmann psychrometer
    using Sprung's [1]_ formula (as seen on wikipedia)

    Parameters
    ----------
    t_dry : float or np.array of floats
        dry temperature [deg C]
    t_wet : float or np.array of floats
        wet temperature [deg C]
    p : float or np.array of floats or None, optional
        air pressure [hPa], if None: use the simplified version of the formula,
        default is None

    Returns
    -------
    e : float or np.array of floats
        vapour pressure [hPa]

    Notes
    -----
    If p is None, a simplification is used which can be used below 500m above
    sea level

    References
    ----------
    .. [1] Sprung, A.: Ueber die Bestimmung der Luftfeuchtigkeit mit Hilfe des
        Assmannschen Aspirationspsychrometers, Z. Angew. Meteorol., Das Wetter,
        5 (1888), S. 105?108

    Examples
    --------
    >>> psychro2e(np.array([17.5,18.9]),np.array([12.3,12.1]))
    array([10.82619823,   9.56701424])

    >>> p = np.array([800,800])
    >>> psychro2e(np.array([17.5,18.9]),np.array([12.3,12.1]),p=p)
    array([11.57630294,  10.54309631])
    """
    p = np.array(p)
    if p.any():
        cp = 1005.4  # specific heat capacity of air J/(kg*K)
        mu = 0.622  # molar mass ratio water/air
        # latent heat of vaporization of water [J/kg]
        lam = (
            -0.0000614342 * t_dry**3
            + 0.00158927 * t_dry**2
            - 2.36418 * t_dry
            + 2500.79
        ) * 1000
        gamma = p * cp / (mu * lam)  # hPa/K
    else:
        gamma = 0.67  # hPa/K, simplification, can be used below 500m asl
    return sat_vap_p(t_wet) - gamma * (t_dry - t_wet)


def slope_sat_p(at):
    """slope of the saturation vapor pressure function,
    depends only on air temperature

    Parameters
    ----------
    at : float or numpy.array of floats
        (mean daily) air temperature [deg C]

    Returns
    -------
    slope : float or numpy.array of floats
        slope of the saturation vapor pressure function  [kPa/deg C]

    Notes
    -----
    A polynomial is used to evaluate the slope and is only valid
    for -5 < 'at' > 45 [deg C].

    References
    ----------
    .. [1] Campbell Scientific (1995) application note 4-D: On-Line
        Estimation of Grass Reference Evapotranspiration with the Campbell
        Scientific Automated Weather Station
    """

    slope = (45.3 + 2.97 * at + 0.0549 * at**2 + 0.00223 * at**3) / 1000

    return slope


def esi(at, rh, sw):
    """Environmental stress index (wet bulb globe temperature substitute)

    Parameters
    ----------
    at : float or numpy.array of floats
        air temperature [deg C]
    rh : float or numpy.array of floats
        relative humidity [%]
    sw : float or numpy.array of floats
        solar radiation [W / m²]

    Returns
    -------
    ESI : environmental stress index

    References
    ----------
    .. [1] Moran et al., “An Environmental Stress Index (ESI) as a
       Substitute for the Wet Bulb Globe Temperature (WBGT).”

    """
    return (
        0.63 * at
        - 0.03 * rh
        + 0.002 * sw
        + 0.0054 * (at * rh)
        - 0.073 / (0.1 + sw)
    )


def wet_bulb_stull(at, rh):
    """
    Examples:
    ---------
    >>> wet_bulb_stull(20, 50)
    13.7
    """
    return (
        at * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(at + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh) ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def humidex(at, rh):
    dew = dewpoint(at, rh=rh / 100)
    return at + 0.5555 * (
        6.11 * np.exp(5417.753 * (1 / 273.15 - 1 / (273.15 + dew))) - 10
    )


def apparent_temperature(Ta, rh, ws, Q):
    """Australian Bureau of Meteorology formulation of Steadman (1984)."""
    e = rh / 100 * 6.105 * np.exp(17.27 * Ta / (237.7 + Ta))
    return Ta + 0.348 * e - 0.7 * Q / (ws + 10) - 4.25


def brunner_compound(sti, spi, sequential=False, progress=False):
    """Rank-based hot-dry index.


    Notes
    -----
    Brunner 2021 uses E-GPD for precipitation and a STI index for
    temperature. This implementation just uses empirical ranks.

    References
    ----------
    Brunner, Manuela I., Eric Gilleland, and Andrew W. Wood.
    “Space–Time Dependence of Compound Hot–Dry Events in the United
    States: Assessment Using a Multi-Site Multi-Variable Weather
    Generator.” Earth System Dynamics 12, no. 2 (May 19, 2021):
    621–34. https://doi.org/10.5194/esd-12-621-2021.
    """
    if progress:
        from tqdm import tqdm as progress
    else:

        def progress(x, *args, **kwds):
            return x

    assert len(sti) == len(spi)
    sti_ranks = bottleneck.nanrankdata(sti, axis=-1)
    spi_ranks = bottleneck.nanrankdata(-spi, axis=-1)
    try:
        if sequential:
            raise MemoryError
        bc = np.mean(
            (sti_ranks[..., None, :] <= sti_ranks[..., :, None])
            & (spi_ranks[..., None, :] <= spi_ranks[..., :, None]),
            axis=-1,
        )
        nan_mask = np.isnan(sti_ranks) | np.isnan(spi_ranks)
        bc[nan_mask] = np.nan
    except MemoryError:
        shape_before = sti_ranks.shape
        sti_ranks = sti_ranks.reshape(-1, shape_before[-1])
        spi_ranks = spi_ranks.reshape(-1, shape_before[-1])
        bc = np.full_like(sti_ranks, np.nan)
        for row_i in progress(range(len(sti_ranks)), total=len(sti_ranks)):
            for i, (sti_rank, spi_rank) in enumerate(
                zip(sti_ranks[row_i], spi_ranks[row_i])
            ):
                if np.isnan(sti_rank) or np.isnan(spi_rank):
                    continue
                sti_row = sti_ranks[row_i]
                spi_row = spi_ranks[row_i]
                finite_mask = np.isfinite(sti_row) & np.isfinite(spi_row)
                if np.any(finite_mask):
                    bc[row_i, i] = np.mean(
                        (sti_row <= sti_rank) & (spi_row <= spi_rank),
                        # (sti_row[finite_mask] <= sti_rank)
                        # & (spi_row[finite_mask] <= spi_rank),
                        axis=-1,
                    )
        bc = bc.reshape(shape_before)
    return bc


def _measure_ar(
    varname, distribution, agg_func_name, obs_ar, weeks=1, reference=None
):
    obs_ar = obs_ar.transpose(..., "time").rolling(time=weeks * 7)
    obs_ar = getattr(obs_ar, agg_func_name)()
    if reference is None:
        reference = obs_ar
    else:
        reference = reference.rolling(time=weeks * 7)
        reference = getattr(reference, agg_func_name)()
    months = reference.time.dt.month
    param_names = [param.name for param in distribution._param_info()]
    param_coords = {key: obs_ar.coords[key] for key in obs_ar.dims}
    param_coords["param"] = param_names
    params = xr.DataArray(
        np.empty(obs_ar.shape + (len(param_names),)),
        coords=param_coords,
    )
    for month, monthly_data in reference.groupby(months):
        flat = monthly_data.data.reshape(-1, monthly_data.sizes["time"])
        params_flat = np.array(
            [distribution.fit(row[np.isfinite(row)]) for row in flat]
        )
        params.loc[dict(time=(months == month))] = params_flat.reshape(
            monthly_data.shape[:-1] + (1, len(param_names))
        )
    measure_data = stats.norm.ppf(
        distribution.cdf(
            obs_ar,
            **{
                key: val.data
                for key, val in params.to_dataset("param").items()
            },
        )
    )
    measure_coords = obs_ar.coords.copy()
    measure_coords["variable"] = varname
    # Explicitly specify dims to avoid relying on coord ordering
    # squeeze() removes the 'variable' dimension, so dims are obs_ar.dims without 'variable'
    measure_dims = [dim for dim in obs_ar.dims if dim != "variable"]
    measure_ar = xr.DataArray(
        np.squeeze(measure_data), coords=measure_coords, dims=measure_dims
    )
    return measure_ar


def STI_ds(obs_ds, weeks=1, reference=None):
    sti_ar = _measure_ar(
        "sti",
        stats.norm,
        "mean",
        obs_ds.to_dataarray("variable"),
        weeks=weeks,
        reference=reference,
    )
    return xr.Dataset(dict(sti=sti_ar))


def STI_ar(obs_ar, weeks=1, reference=None):
    return _measure_ar(
        "sti",
        stats.norm,
        "mean",
        obs_ar,
        weeks=weeks,
        reference=reference,
    )


def SPI_ds(obs_ds, weeks=1, reference=None):
    spi_ar = _measure_ar(
        "spi",
        stats.genpareto,
        "sum",
        obs_ds.to_dataarray("variable"),
        weeks=weeks,
        reference=reference,
    )
    return xr.Dataset(dict(spi=spi_ar))


def SPI_ar(obs_ar, weeks=1, reference=None):
    return _measure_ar(
        "spi",
        stats.genpareto,
        "sum",
        obs_ar,
        weeks=weeks,
        reference=reference,
    )


if __name__ == "__main__":
    # doys = vg.times.datetime2doy(vg.times.str2datetime(500 * ["2011-09-28T11:27"]))
    # # pot_s_rad(doys)
    # sunshine_pot(doys)
    # from datetime import date
    # sunshine_riseset(date(2018, 6, 1), 8.848, 48.943)

    # import doctest

    # doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

    from vg.meteo import dwd_opendata

    theta = dwd_opendata.load_data(
        ("Stötten", "Konstanz"), "temperature", time="hourly"
    ).to_dataset("station")
    theta = (
        theta.interpolate_na("time")
        .transpose(..., "time")
        .squeeze()
        .resample(time="1d")
        .sum()
        .sel(time=slice("1990", "2020"))
        .dropna("time")
    )
    sti = STI_ds(theta, weeks=6)

    # prec = dwd_opendata.load_data(
    #     ("Stötten", "Konstanz"), "precipitation", time="hourly"
    # ).to_dataset("station")
    # prec = (
    #     prec.interpolate_na("time")
    #     .transpose(..., "time")
    #     .squeeze()
    #     .resample(time="1d")
    #     .sum()
    #     .sel(time=slice("1990", "2020"))
    #     .dropna("time")
    # )
    # spi = SPI_ds(prec, weeks=6)
