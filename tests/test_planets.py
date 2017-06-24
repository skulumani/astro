"""test all the functions in planets 
"""
import numpy as np

from astro import planets
import pdb


rtol = 1e-3

def test_sun_earth_eci_usafa():
    jd = 2453905.50000000
    expected_rsun = [6371243.918400, 139340081.269385, 60407749.811252]
    rsun, ra, dec = planets.sun_earth_eci(jd)
    np.testing.assert_allclose(rsun, expected_rsun, rtol=1e-4)


class TestMercuryCOE():
    jd = 2457928.5

    coe, _, _, _, _ = planets.planet_coe(jd, 0)
    
    def test_mercury_p(self):
        pexp = 3.870982978406551e-1 * (1 - 2.06334003441381e-1**2)
        np.testing.assert_allclose(self.coe.p, pexp, rtol=rtol)

    def test_mercury_ecc(self):
        np.testing.assert_allclose(self.coe.ecc, 2.056334003441381e-1, rtol=rtol)
