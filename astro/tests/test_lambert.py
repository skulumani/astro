"""Test out the lambert solver
"""

import numpy as np

from astro import lambert, constants

class TestValladoExample():
    """Example 7-5 from Vallado
    """

    r1 = np.array([15945.34, 0, 0])
    r2 = np.array([12214.83899, 10249.46731, 0])
    tof = 76 * 60
   
    def test_universal_method(self):
        v1_expected = np.array([2.058913, 2.915965, 0])
        v2_expected = np.array([-3.451565, 0.910315, 0])
        v1, v2 = lambert.lambert_universal(self.r1, self.r2, 'short', 0,
                                           self.tof, constants.earth.mu,
                                           constants.earth.radius)
        np.testing.assert_allclose((v1, v2), (v1_expected, v2_expected),
                                   rtol=1e-5)

    def test_minimum_energy(self):
        v1_expected = np.array([2.047409, 2.924003, 0])
        v2_expected = np.array([-3.447919, 0.923867, 0])
        tof_expected = 75.6708 * 60
        a_expected = 10699.48385
        v1, v2, tof, a, p, ecc = lambert.lambert_minenergy(self.r1, self.r2,
                                                           constants.earth.radius,
                                                           constants.earth.mu,
                                                           'short')
        np.testing.assert_allclose(v1, v1_expected, rtol=1e-5)
        np.testing.assert_allclose(v2, v2_expected, rtol=1e-5)
        np.testing.assert_allclose(tof, tof_expected, rtol=1e-5)
        np.testing.assert_allclose(a, a_expected)

