"""
Do lots of functions for the planets, sun and/or moon
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from ..kinematics import attitude

def sun_earth_eci(jd):
    """Calcualte the sun vector in the geocentric inertial frame
    """
    twopi = 2 * np.pi
    rad = np.pi/180
    
    # initialize
    N = JD - 2451545.0

    meanlong = 280.461 + 0.9856476*N
    meanlong = attitude.normalize(meanlong, 0, 360)

    meananomaly = 357.528 * 0.9856003 * N
    meananomaly = attitude.normalize(meananomaly * rad, 0, twopi)

    if meananomaly < 0:
        meananomaly = twopi + meananomaly




