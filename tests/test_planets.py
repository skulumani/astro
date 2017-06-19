"""test all the functions in planets 
"""
import numpy as np

from astro import planets
import pdb

def test_sun_earth_eci_vallado():
    jd = 2453827.5
    expected_rsun = [146186178.0792,28789122.463592, 12481127.00722]
    expected_ra = 11.140899 * np.pi/180
    expected_dec = 4.788425 * np.pi/180
    rsun, ra, dec = planets.sun_earth_eci(jd)
    np.testing.assert_array_almost_equal(rsun, expected_rsun, decimal=0)

