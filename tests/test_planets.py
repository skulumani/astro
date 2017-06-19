"""test all the functions in planets 
"""
import numpy as np

from astro import planets
import pdb

def test_sun_earth_eci_usafa():
    jd = 2453905.50000000
    expected_rsun = [6371243.918400, 139340081.269385, 60407749.811252]
    rsun, ra, dec = planets.sun_earth_eci(jd)
    np.testing.assert_allclose(rsun, expected_rsun,rtol=1e-4)

