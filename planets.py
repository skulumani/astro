"""
Do lots of functions for the planets, sun and/or moon
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys 
sys.path.append('..')
import numpy as np
from tle_predict.kinematics import attitude
import pdb

def sun_earth_eci(jd):
    """Calcualte the sun vector in the geocentric inertial frame
    """
    twopi = 2 * np.pi
    deg2rad = np.pi/180
    Tut1 = ( jd - 2451545.0) / 36525

    meanlong = attitude.normalize(280.460 + 36000.771 * Tut1, 0, 360.0)

    Ttdb = Tut1 # approximation
    meananomaly = attitude.normalize(357.5277233 + 35999.05034 * Ttdb, 0, 360)
    meananomaly = meananomaly * deg2rad
    eclplong = meanlong + 1.914666471 * np.sin(meananomaly) + 0.019994643 * (
            np.sin(2 * meananomaly))
    eclplong = attitude.normalize(eclplong, 0, 360)
    obliquity = attitude.normalize(23.439291 - 0.0130042 * Ttdb, 0, 360)
    # convert everything to radians
    eclplong = eclplong * deg2rad 
    obliquity = obliquity * deg2rad
    meanlong = meanlong * deg2rad
    # right ascension and declination
    dec = np.arcsin(np.sin(obliquity) * np.sin(eclplong))

    ra = np.arctan2(np.cos(obliquity)*np.sin(eclplong)/np.cos(dec), 
            np.cos(eclplong)/np.cos(dec))
    # magnitude of sun vector 
    rsun_mag = 1.000140612 - 0.016708617 * np.cos(meananomaly) - (
            0.000139589 * np.cos(2 * meananomaly))
    rsun = rsun_mag * np.array([np.cos(eclplong), np.cos(obliquity)*np.sin(eclplong),
        np.sin(obliquity)*np.sin(eclplong)])

    rsun = np.squeeze(rsun * 149597870.7) # output in km
    return rsun, ra, dec
