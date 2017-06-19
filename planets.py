"""
Do lots of functions for the planets, sun and/or moon
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from kinematics import attitude
import pdb

twopi = 2 * np.pi
deg2rad = np.pi/180
rad2deg = 180 / np.pi
au2km = 149597870.0

def sun_earth_eci(jd):
    """This function calculates the Geocentric Equatorial position vector for
    the Sun given the Julian Date.  This is the low precision formula and is
    valid for years from 1950 to 2050.  Accuaracy of apparent coordinates is
    0.01 degrees.  Notice many of the calculations are performed in degrees,
    and are not changed until later.  This is due to the fact that the Almanac
    uses degrees exclusively in their formulations.

    Algorithm     : Calculate the several values needed to find the vector
                    Be careful of quadrant checks

    Author        : Capt Dave Vallado  USAFA/DFAS  719-472-4109  25 Aug 1988
    In Ada        : Dr Ron Lisowski    USAFA/DFAS  719-472-4110  17 May 1995
    In MatLab     : Dr Ron Lisowski    USAFA/DFAS  719-333-4109  10 Oct 2001
    In Python     : Shankar Kulumani   GWU         630-336-6257  19 Jun 2017

    Inputs        :
    JD          - Julian Date                            days from 4713 B.C.

    Outputs       :
    RSun        - IJK Position vector of the Sun         km
    RtAsc       - Right Ascension                        rad
    Decl        - Declination                            rad

    Locals        :
    MeanLong    - Mean Longitude
    MeanAnomaly - Mean anomaly
    N           - Number of days from 1 Jan 2000
    EclpLong    - Ecliptic longitude
    Obliquity   - Mean Obliquity of the Ecliptic

    Constants     :
    Pi          -
    TwoPI       -
    InvRad      - Radians per degree

    Coupling      :

    References             :
    1996 Astronomical Almanac Pg. C24
    http://aa.usno.navy.mil/faq/docs/SunApprox.php
    """
    N = jd - 2451545.0
    
    meanlong = 280.461 + 0.9856474 * N
    meanlong = attitude.normalize(meanlong, 0, 360)

    meananomaly = 357.528 + 0.9856003 * N
    meananomaly = attitude.normalize(meananomaly * deg2rad, 0, twopi)
    if meananomaly < 0:
        meananomaly = twopi + meananomaly

    eclplong = meanlong + 1.915 * np.sin(meananomaly) + 0.020 * np.sin(2 * meananomaly)
    obliquity = 23.439 - 0.0000004 * N

    meanlong = meanlong * deg2rad
    if meanlong < 0:
        meanlong = twopi + meanlong

    eclplong = eclplong * deg2rad
    obliquity = obliquity * deg2rad

    ra = np.arctan2(np.cos(obliquity) * np.sin(eclplong), np.cos(eclplong))
    dec = np.arcsin(np.sin(obliquity) * np.sin(eclplong))

    # equation of time
    eqtime = meanlong * rad2deg / 15 - ra * rad2deg / 15

    # sun vector
    sun_dist = 1.00014 - 0.01671 * np.cos(meananomaly) - 0.00014 * np.cos(2 * meananomaly)
    
    semidiameter = 0.2666 / sun_dist # angular semidiamter in deg
    sun_eci = [np.cos(eclplong) * sun_dist * au2km, 
               np.cos(obliquity) * np.sin(eclplong) * sun_dist * au2km, 
               np.sin(obliquity) * np.sin(eclplong) * sun_dist * au2km]
    
    return np.squeeze(sun_eci), ra, dec
