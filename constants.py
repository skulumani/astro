"""Astrodynamic constants

This module will load a wide variety of WGS84 derived constants as module
attributes. These constants may be referenced in subsequent scripts and functions.
These variables are case specific and must be noted.

Notes
-----

To use these constants in your scripts/functions, simply import this module

from astro import constants

Then use any of the attributes as:

deg2rad = constants.deg2rad

Attributes
----------
constants : module level attributes broken down by planetary body
    rot_per : axial rotational period in rev/day
    radius : mean equatorial radius in km
    mu : graviational parameter in km^3/sec^2
    orbit_sma : semi-major axis of orbit in km
    orbit_per : period of orbit in sec
    orbit_ecc : eccentricity of orbit
    orbit_inc : inclination of orbit in deg
    mass : body mass (derived from mu and G) in kg
    G : universal graviational constant in km^3/kg*sec^2

Author
------
Shankar Kulumani 1 Sept 2012
    list revisions
Shankar Kulumani 19 Sept 2012
    added distance conversions
    added time conversions
Shankar Kulumani 30 Oct 2012
    added semiparamter to constants
Shankar Kulumani 21 June 2017
    Created for MAE3145 in Python

References
----------
Astro 321 USAFA 2007
AAE 532 Purdue 2012 planetary_constants.pdf (from AAE532 supplements)

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
import numpy as np

# HALFPI,PI2           PI/2, & 2PI in various names
halfpi = np.pi / 2.0
twopi = 2.0 * np.pi

zero_inc_ecc = 0.015                # Small number for incl & ecc purposes
small = 1.0E-6                  # Small number used for tolerance purposes
undefined = 999999.1

# General constants
deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

G = 6.673e-20
au2km = 149597870.0  # km
km2au = 1 / au2km
km2re = 1 / 6378.137  # km to Earth radii
re2km = 6378.137

sec2hr = 1 / 3600
hr2sec = 3600
day2sec = 86400.0                   # sec/day
sec2day = 1 / 86400

# Earth Characteristics from WGS 84
obliquity = np.radians(23.43928)

#  Moon & Sun Characteristics from WGS 84
mu_moon = 4902.774191985            # km^3/sec^2
mu_sun = 1.32712438e11              # km^3/sec^2

# Planetary constants
# namedtuple to hold constants for each body
Body = namedtuple('Body', ['rot_per', 'radius', 'mu', 'mass', 'orbit_sma',
                           'orbit_per', 'orbit_ecc', 'orbit_inc', 'p'])

Earth = namedtuple('Earth', ['rot_per', 'radius', 'mu', 'mass', 'orbit_sma',
                             'orbit_per', 'orbit_ecc', 'orbit_inc', 'p',
                             'omega', 'sidepersol', 'radperday', 'flat',
                             'eesqrd', 'ee', 'J2', 'J3', 'J4'])
# earth
earth = Earth(rot_per=1.0027576,
              radius=6378.137,
              mu=398600.5,
              orbit_sma=1.49589800e8,
              orbit_per=31555647.16,
              orbit_ecc=.01671020,
              orbit_inc=4.98816000e-5,
              mass=398600.5 / G,
              p=1.49589800e8 * (1 - .01671020**2),
              omega=0.000072921151467,     # rad/sec
              sidepersol=1.00273790935,          # Sidereal Days/Solar Day
              radperday=6.30038809866574,        # rad/day for the Earth
              flat=1.0 / 298.257223563,            # Earth flattening factor
              # Earth eccentricty squared
              eesqrd=(2.0 - (1.0 / 298.257223563)) * (1.0 / 298.257223563),
              ee=8.1819190842622e-2,
              J2=0.00108263,                     # Zonal Harmonic perturbation
              J3=-0.00000254,
              J4=-0.00000161)

# define sun constants
sun = Body(rot_per=0.0394011,
           radius=695990.0,
           mu=1.32712000e11,
           mass=1.32712000e11 / G,
           orbit_sma=0,
           orbit_per=0,
           orbit_ecc=0,
           orbit_inc=0,
           p=0)


# define moon constants
moon = Body(rot_per=.0366004,
            radius=1737.50,
            mu=4.902800e3,
            orbit_sma=3.84400000e5,
            orbit_per=2360592,
            orbit_ecc=.05540000,
            orbit_inc=5.16000,
            mass=4.902800e3 / G,
            p=3.84400000e5 * (1 - .05540000**2))

# mercury
mercury = Body(rot_per=.0170515,
               radius=2.439000e3,
               mu=2.203210e4,
               orbit_sma=5.79092000e7,
               orbit_per=7600568.601,
               orbit_ecc=0.205631,
               orbit_inc=7.00487,
               mass=2.203210e4 / G,
               p=5.79092000e7 * (1 - 0.205631**2))
# venus
venus = Body(rot_per=.0041149,
             radius=6051.80,
             mu=3.24859000e5,
             orbit_sma=1.0820900e8,
             orbit_per=19414191.77,
             orbit_ecc=.006773,
             orbit_inc=3.39471,
             mass=3.24859000e5 / G,
             p=1.0820900e8 * (1 - .006773**2))

# mars
mars = Body(
    rot_per=.9746985,
    radius=3397.00,
    mu=4.282840e4,
    orbit_sma=2.27937000e8,
    orbit_per=59353583.28,
    orbit_ecc=.09341230,
    orbit_inc=1.85061,
    mass=4.282840e4 / G,
    p=2.27937000e8 * (1 - .09341230**2))

# jupiter
jupiter = Body(
    rot_per=2.4181458,
    radius=71492.00,
    mu=1.26687000e8,
    orbit_sma=7.7841200e8,
    orbit_per=374396573,
    orbit_ecc=.04839270,
    orbit_inc=1.30530,
    mass=1.26687000e8 / G,
    p=7.7841200e8 * (1 - .04839270**2))

callisto = Body(
    rot_per=0,
    radius=0,
    mu=7179.29,
    orbit_sma=1.883e6,
    orbit_per=0,
    orbit_ecc=0,
    orbit_inc=0,
    mass=0,
    p=0)

#  saturn
saturn = Body(
    rot_per=2.2522523,
    radius=60330.00,
    mu=3.79313000e7,
    orbit_sma=1.42673000e9,
    orbit_per=929341659.8,
    orbit_ecc=.05415060,
    orbit_inc=2.48446,
    mass=3.79313000e7 / G,
    p=1.42673000e9 * (1 - .05415060**2))


titan = Body(
    rot_per=0.627393,
    radius=2575.5,
    mu=8978.19,
    orbit_sma=1221830.0,
    orbit_per=15.94542*day2sec,
    orbit_ecc=0.0292,
    orbit_inc=0.33,
    mass=8978.19 / G,
    p=1221830.0 * (1 - 0.2**2))

# uranus
uranus = Body(
    rot_per=1.3921178,  # retrograde rotation
    radius=26200.00,
    mu=5.79397000e6,
    orbit_sma=2.87097000e9,
    orbit_per=2653128427,
    orbit_ecc=.04716770,
    orbit_inc=.76986,
    mass=5.79397000e6 / G,
    p=2.87097000e9 * (1 - .04716770**2))

# neptune
neptune = Body(
    rot_per=1.4897579,
    radius=25225.00,
    mu=6.835110e6,
    orbit_sma=4.49825000e9,
    orbit_per=5203301252,
    orbit_ecc=.00858587,
    orbit_inc=1.76917,
    mass=6.835110e6 / G,
    p=4.49825000e9 * (1 - .00858587**2))

# pluto
pluto = Body(
    rot_per=.1565631,
    radius=1195.00,
    mu=8.737670e2,
    orbit_sma=5.906638e9,
    orbit_per=7829522968,
    orbit_ecc=.24880800,
    orbit_inc=17.14180,
    mass=5.906638e9 / G,
    p=5.906638e9 * (1 - .24880800**2))
