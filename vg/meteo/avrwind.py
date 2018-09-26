""" written by Raphael Lutz 2010
"""
from __future__ import print_function
from __future__ import division

from builtins import range
from past.utils import old_div
import numpy as np
import datetime
from vg import times


def angle2component(angle, norm, wind=True):
    """Converts wind data with the format (direction, speed) into the velocity
    components (u,v).

    If ``wind`` == True, 0 degree is wind blowing from north, 90 degree is wind
    from east, etc. If False (if data is not wind data but ship movement or
    anything else), 0 degree is pointing to north, 90 to east etc.

    Parameters
    ----------
    angle : float or np.array of floats
        direction [0,360] where the wind is coming from
    norm : float or np.array of floats
        wind speed
    wind : boole, optional
        input data is wind data, default is True

    Returns
    -------
    u : float or np.array of floats
        component u pointing from west to east
    v : float or np.array of floats
        component v pointing from south to north


    See Also
    --------
    component2angle : vice versa

    Examples
    --------
    >>> angle2component(90, 1)
    (-1.0, -6.123233995736766e-17)

    >>> angle2component([90,180],[1,1], wind=False)
    (array([  1.00000000e+00,   1.22464680e-16]),
     array([  6.12323400e-17,  -1.00000000e+00]))
     """
    try:  # look if single value or vector
        _ = angle[0]
        angle, norm = np.array(angle), np.array(norm)
    except:
        pass
    angle = angle / 360.0 * 2 * np.pi
    u = np.sin(angle) * norm
    v = np.cos(angle) * norm
    if wind:
        u, v = -u, -v
    return u, v


def component2angle(u, v, wind=True):
    """
    converts a tuple of (u,v) to (angle,norm) in deg 0=north 90=east

    Parameters
    ----------
    u : float or np.array of floats
        component u pointing from west to east
    v : float or np.array of floats
        component v pointing from south to north
    wind : boole, optional
        input data is wind data, default is True

    Returns
    -------
    angle : float or np.array of floats
        direction [0,360] where the wind is coming from
    norm : float or np.array of floats
        wind speed

    See Also
    --------
    angle2component : vice versa

    Examples
    --------
    >>> component2angle(1,0)
    (270.0, 1.0)

    >>> component2angle([1,-1,0,0],[0,0,1,-1], wind=False)
    (array([  90.,  270.,    0.,  180.]), array([ 1.,  1.,  1.,  1.]))
    """
    try:
        _ = u[0]  # test if input is single value or vector
        u, v = np.array(u), np.array(v)
    except:
        pass

    try:
        if len(u) != len(v):
            raise ValueError("Error:length of u,v differ")
    except TypeError:
        pass
    if wind:
        u, v = -u, -v
    # for entertainment, take a look into the docstring of np.arctan2...
    # (yes u and v are switched, yes we look at a different angle, ...)
    angle_rad = np.arctan2(u, v)
    # normalize to (0,2pi]
    angle_rad = (angle_rad + 2 * np.pi) % (2 * np.pi)
    angle = np.degrees(angle_rad)
    norm = (u ** 2 + v ** 2) ** .5
    return angle, norm


def avrwind_vec(u, v, sec=16, verbose=True, timeinfo=None,
                return_secdata=False, wind=True):
    """
    calculates the average windspeed and direction using vector averaging

    Parameters
    ----------
    u : float or np.array of floats
        component u pointing from west to east
    v : float or np.array of floats
        component v pointing from south to north
    sec, verbose, timeinfo, return_secdata : kwargs
        kwargs for avrwind_sec, not used here. Only for easier making the
        function call of the two functions the same
    wind : boole, optional
        input data is wind data, default is True (for details: see
        documentation to angle2component)

    Returns
    -------
    avr_u : float
        averaged u velocity
    avr_v : float
        averaged v velocity
    angle_avr : float
        direction of averaged wind
    norm_avr : float
        speed of averaged wind
    norm_uv_direct : float
        speed: averaged input wind speeds

    See Also
    --------
    avrwind_sec : wind averaging using sector method
    avrwind : convienience function

    Examples
    --------
    >>> avrwind_vec([1,23],[-1,5])
    (12.0, 2.0, 260.53767779197437, 12.165525060596439, 12.475709077126368)
    """
    u, v = np.array(u), np.array(v)
    if len(u) != len(v):
        print("ERROR: mismatch -> length of u,v differ!")
        return(None)
    norm_uv_direct = np.average(np.sqrt(u ** 2 + v ** 2))

    avr_u, avr_v = np.average(u), np.average(v)
    angle_dir = component2angle(avr_u, avr_v, wind=wind)
    return avr_u, avr_v, angle_dir[0], angle_dir[1], norm_uv_direct


def avrwind_sec(u, v, sec=16, verbose=True, timeinfo=None,
                return_secdata=False, wind=True):
    """
    function to calculate the average windspeed and directions via sectors

    looks for sector with most wind, gives the sector as dir and the average of
    wind speed in this sector as speed

    Parameters
    ----------
    u : float or np.array of floats
        component u pointing from west to east
    v : float or np.array of floats
        component v pointing from south to north
    sec : int, optional
        number of sectors used for interpolation. Default is 16
    verbose : {True, False}, optional
        give information about problems during calculation
    timeinfo : None or string, optional
        time string, fmt="%d.%m.%Y %H:%M:%S", only for problem information
    return_secdata : {False, True}, optional
        print information about sectors to screen
    wind : boole, optional
        input data is wind data, default is True (for details: see
        documentation to angle2component)

    Returns
    -------
    avr_u : float
        averaged u velocity
    avr_v : float
        averaged v velocity
    avr_wind_direction : float
        most common wind direction
    average_speed_maxsector : float
        average speed in most commen wind direction
    norm_uv_direct : float
        speed: averaged input wind speeds

    See Also
    --------
    avrwind_vec : wind averaging using vector method
    avrwind : convienience function

    Examples
    --------
    >>> avrwind_sec([1,10,10],[-1,5,5])
    (10.329287188579819, 4.2785308431564255, 247.5, 11.180339887498949, 0.6666666666666666, 7.924964445790331)

    >>> avrwind_sec([1,10,10],[-1,5,5], wind=False)
    (10.329287188579821, 4.2785308431564202, 67.5, 11.180339887498949, 0.6666666666666666, 7.924964445790331)
    """
    sector_means = np.array([(i * float(360) / sec) for i in range(sec + 1)])
    hits_averages = np.zeros((2, sec))

    angles, velocities = component2angle(np.array(u), np.array(v), wind=wind)
    norm_uv_direct = np.average(velocities)

    sectors = np.array([np.argmin(np.abs(sector_means - an)) for an in angles])
    sectors[np.where(sectors == sec)] = 0  # remove doubled sector
    for m in range(len(u)):
        hits_averages[0][int(sectors[m])] += 1  # adds a hit in the sector
        # + average windsp to the sec
        hits_averages[1][int(sectors[m])] += velocities[m]

    max_hits = int(hits_averages.max(axis=1)[0])  # number of most hits
    max_sec = hits_averages.argmax(axis=1)[0]   # sectornumber of maximum sec
#     max_speed_sec = hits_averages.argmax(axis=1)[1]  # sectornr max speed

    # test if there is sector with equal hits as maxsec
    test = hits_averages[0].copy()
    test.sort()
    if test[-1] == test[-2] and verbose == True:
        try:
            print(timeinfo + "  Caution multiple sectors with equal hits \n")
            return_secdata = True
        except:
            pass  # if no timeinfo?

    elif old_div(test[-2], test[-1]) > 0.75 and verbose == True:
        try:
            print(timeinfo + "  2nd most direction alike numerous as first \n")
            return_secdata = True
        except:
            pass

    average_speed_maxsector = old_div(hits_averages[1][max_sec], float(max_hits))

    avr_wind_direction = sector_means[max_sec]
    rel_freq_mainsector = (old_div(max_hits, float(len(u))))

    #nullabfrage? kein error
    if (old_div(average_speed_maxsector, norm_uv_direct) < 0.5 or
        old_div(average_speed_maxsector, norm_uv_direct) > 2) and verbose == True:
        print(timeinfo + "  Caution average windspeed in main sector and " + \
              "overall windspeeds differ extreme \n ")
        # what would be a good value?

    if return_secdata:
        # print sectors, counts, and sum of wind speed in sectors
        print((sector_means[:-1], '\n', list(hits_averages[0]),
               '\n', list(hits_averages[1])))

    u_v = angle2component(avr_wind_direction, average_speed_maxsector,
                          wind=wind)

    return (u_v[0], u_v[1], avr_wind_direction, average_speed_maxsector,
            rel_freq_mainsector, norm_uv_direct)


#-----------------------------------------------------------------------------#
#   main function                                                             #
#-----------------------------------------------------------------------------#
def avrwind(u, v, date_time, new_timeres, method='vector', verbose=True,
            sec=16, wind_=True):
    """
    function to convert measured data from one time resolution to another
    you can decide which method you will use, sector=sector-wise or
    vector=vector-adding polygon trace

    Parameters
    ----------
    u : float or np.array of floats
        component u pointing from west to east
    v : float or np.array of floats
        component v pointing from south to north
    date_time : timeinfo as list/array of string '01.01.2010 12:30:40'
        or datetime-object, or single timedelta in seconds(e.g: 60)
    new_timeres : int
        [seconds] new time resolution of output data
    method : {'vector','sector'}
    verbose : {True, False}, optional
        give information about problems during calculation
    sec : int, optional
        number of sectors used for sector interpolation. Default is 16
    wind_ : boole, optional
        input data is wind data, default is True (for details: see
        documentation to angle2component)

    Returns
    -------
    avr_u : float
        averaged u velocity
    avr_v : float
        averaged v velocity
    avr_wind_direction : float
        direction of averaged wind ('vector') or most common wind direction
        ('sector')
    average_speed_maxsector : float
        speed of averaged wind ('vector') or average speed in most commen wind
        direction ('sector')
    return_time : list or array if str or datetime objects, depending in input

    See Also
    --------
    avrwind_vec : wind averaging using vector method
    avrwind_sec : wind averaging using sector method

    Examples
    --------
    >>> date_str = ['01.01.2010 12:00:00', '01.01.2010 12:01:00']
    >>> avrwind([1,2], [3,4], date_str, 60*2, 'vector')
    [[1.5],
     [3.5],
     [203.19859051364818],
     [3.8078865529319543],
     array(['01.01.2010 12:01:00'],
          dtype='|S19')]

    >>> avrwind([1,2], [3,4], date_str, 60*2, 'vector', wind_=False)
    [[1.5],
     [3.5],
     [23.198590513648185],
     [3.8078865529319543],
     array(['01.01.2010 12:01:00'],
          dtype='|S19')]

    >>> avrwind([1,2], [3,4], 60, 120, 'sector', sec=16)
    [[1.4607818031736237],
     [3.526639240889589],
     [202.5],
     [3.8172068075839798],
     [datetime.datetime(1900, 1, 1, 0, 1)]]
    """

    if len(u) != len(v):
        print("Error: different length in vectors u, v")
        return
    #if only time span is given: create a time vector beginning 1.1.1900 0:0:0
    if type(date_time) == type(1):
        date_time = [datetime.datetime(1900, 1, 1) +
                     i * datetime.timedelta(0, date_time)
                     for i in range(len(u))]
        format_ = 'datetime'

    else:
        if  type(date_time[0]) == type(''):
            date_time = times.str2datetime(date_time)
            format_ = "string"
        elif type(date_time[0]) == type(datetime.datetime(1900, 1, 1)):
            format_ = "datetime"
        else:
            print("ERROR: wrong time-info input {arg(3)} only string and \
                    datetime possible")
            return
    # test wheater new timestep < old one
    if new_timeres < (date_time[1] - date_time[0]).seconds:
        print("ERROR: New timeresolution smaller than old one!")
        print("       Can't find values out of nowhere ;)")
        return
    if len(u) != len(date_time):
        print("Error: length of u and timevector differ ")
        # test if timeinfo and time resolution fit
        return

    if method in ('sector', 'sec'):
        method = avrwind_sec
    elif method in ('vector', 'vec'):
        method = avrwind_vec
        sec = None

    else:
        print("ERROR: arg(5) is not a valid method")
        return None

    if type(new_timeres) != type(1):
        print("ERROR arg(4) is not a valid timeresolution [s]")
        return None

    dt_index_low = 0
    time_barrier = date_time[0] + datetime.timedelta(0, new_timeres)
    time_block = [[], []]  # block with vectors u, v which shall be averaged
    return_wind = [[], [], [], []]  # averaged u, v vectors
    return_time = []   # time-vector with new times: middle of averaging time
    i = 0

    while i < len(date_time):  # loop over time-vector
        if date_time[i] < time_barrier:  # creating time-blocks
            time_block[0].append(u[i])
            time_block[1].append(v[i])
            dt_index_high = i
        else:
#            return_time_act = date_time[dt_index_low] + \
#                    (date_time[dt_index_high] - date_time[dt_index_low]) / 2
            return_time_act = date_time[dt_index_high]

            #++++++++++ calling averaging function +++++++++++++++++++++++++++#
            tmp_wind = method(time_block[0], time_block[1], sec, verbose, \
                              times.datetime2str(return_time_act), wind=wind_)
            return_wind[0].append(tmp_wind[0])  # adds averaged date to output
            return_wind[1].append(tmp_wind[1])
            return_wind[2].append(tmp_wind[2])
            return_wind[3].append(tmp_wind[3])
            return_time.append(return_time_act)

            dt_index_low = dt_index_high + 1
            time_barrier = date_time[dt_index_high + 1] + \
                           datetime.timedelta(0, new_timeres)
            time_block = [[], []]
            i = i - 1
        i += 1
    #testing if last block was complete if yes append last data
    #    if not don't and give out caution sign
    if date_time[dt_index_low - 1] + datetime.timedelta(0, new_timeres) == \
                                date_time[dt_index_high] or dt_index_low == 0:
#        return_time_act = date_time[dt_index_low] + \
#                    (date_time[dt_index_high] - date_time[dt_index_low]) / 2
        return_time_act = date_time[dt_index_high]

        tmp_wind = method(time_block[0], time_block[1], sec, verbose, \
                          times.datetime2str(return_time_act), wind=wind_)
        return_wind[0].append(tmp_wind[0])
        return_wind[1].append(tmp_wind[1])
        return_wind[2].append(tmp_wind[2])
        return_wind[3].append(tmp_wind[3])
        return_time.append(return_time_act)

    elif verbose == True:
        print("  Caution: last value(s) lost, didn't cover a full time span \n")

    if format_ == 'string':
        return_time = times.datetime2str(return_time)

    return_list = return_wind
    return_list.append(return_time)

    return return_list

#logging if many values of a time span are missing
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#


def plothist(x, legend):
    import matplotlib.pyplot as plot

    rects1 = plot.bar(color='r')
    rects2 = plot.bar(color='b')

    plot.hist(x, density=True, bins=30)
    plot.legend((rects1[0], rects2[0])(legend[0], legend[1]))
    plot.show()


# next two functions were taken from vg. they are not used there anymore, but
# i hesitate to throw them away
def phi_main(u, v):
    w, ve = np.linalg.eig(np.cov(u, v))
    v_main = ve[:, np.argmax(w)]
    return np.arctan2(v_main[1], v_main[0])


def turn_uv(u, v, phi):
    uv = np.array([u, v]).T
    turn_matrix = np.matrix([[np.cos(phi), -np.sin(phi)],
                             [np.sin(phi), np.cos(phi)]])
    uv = uv * turn_matrix
    u, v = np.asarray(uv).T
    return np.squeeze(u), np.squeeze(v)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
