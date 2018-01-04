"""Test out the lambert solver
"""
import pdb

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
        v1, v2 = lambert.universal(self.r1, self.r2, 'short', 0,
                                   self.tof, constants.earth.mu,
                                   constants.earth.radius)
        np.testing.assert_allclose((v1, v2), (v1_expected, v2_expected),
                                   rtol=1e-5)

    def test_minimum_energy(self):
        v1_expected = np.array([2.047409, 2.924003, 0])
        v2_expected = np.array([-3.447919, 0.923867, 0])
        tof_expected = 75.6708 * 60
        a_expected = 10699.48385
        v1, v2, tof, a, p, ecc = lambert.minenergy(self.r1, self.r2,
                                                   constants.earth.radius,
                                                   constants.earth.mu,
                                                   'short')
        np.testing.assert_allclose(v1, v1_expected, rtol=1e-5)
        np.testing.assert_allclose(v2, v2_expected, rtol=1e-5)
        np.testing.assert_allclose(tof, tof_expected, rtol=1e-5)
        np.testing.assert_allclose(a, a_expected)


class TestEarthMarsTransfer():

    # Earth and SC velocity wrt to sun
    r1 = np.array([150064539.403983, 2267759.896369, -276.754311])
    v1 = np.array([-0.075289, 33.476420, 1.907380])
    # Mars
    r2 = np.array([-207818935.305987, 136713926.487051, 7968576.434594])
    v2 = np.array([-14.189471, -14.839340, -0.833227])

    tof = 210 * 86400
    v1_act, v2_act = lambert.universal(r1, r2, 'short', 0, tof, constants.sun.mu,
                                       constants.sun.radius)
    v1_min, v2_min, tof_min, a, p, ecc = lambert.minenergy(r1, r2,
                                                           constants.sun.radius,
                                                           constants.sun.mu,
                                                           'short')

    def test_v1(self):
        np.testing.assert_allclose(self.v1_act, self.v1, rtol=1e-3)

    def test_v2(self):
        np.testing.assert_allclose(self.v2_act, self.v2, rtol=1e-3)


class TestEarthVenusLongTransfer():

    # Earth and SC velocity wrt to sun
    r1 = np.array([150064539.403983,  2267759.896369, - 276.754311])
    v1 = np.array([23.041751,  0.806102,  4.344444])
    # Venus
    r2 = np.array([108502685.373919,  981095.895427, - 6248799.806056])
    v2 = np.array([34.742706,  0.947442,  4.007763])

    tof = 210 * 86400
    v1_act, v2_act = lambert.universal(r1, r2, 'long', 0, tof, constants.sun.mu,
                                       constants.sun.radius)
    v1_min, v2_min, tof_min, a, p, ecc = lambert.minenergy(r1, r2,
                                                           constants.sun.radius,
                                                           constants.sun.mu,
                                                           'short')

    def test_v1(self):
        np.testing.assert_allclose(self.v1_act, self.v1, rtol=1e-3)

    def test_v2(self):
        np.testing.assert_allclose(self.v2_act, self.v2, rtol=1e-3)

class TestEarthVenusShortTransfer():

    # Earth and SC velocity wrt to sun
    r1 = np.array([-40907159.776698,  -146166176.029225,  2524.752269])
    v1 = np.array([26.151206,  -6.439026,  -0.747977])
    # Venus
    r2 = np.array([21214945.229277,  105654723.068676,  221755.525408])
    v2 = np.array([-36.925365,  8.695987,  1.053181])

    tof = 145 * 86400
    v1_act, v2_act = lambert.universal(r1, r2, 'long', 0, tof, constants.sun.mu,
                                       constants.sun.radius)
    v1_min, v2_min, tof_min, a, p, ecc = lambert.minenergy(r1, r2,
                                                           constants.sun.radius,
                                                           constants.sun.mu,
                                                           'long')

    def test_v1(self):
        np.testing.assert_allclose(self.v1_act, self.v1, rtol=1e-3)

    def test_v2(self):
        np.testing.assert_allclose(self.v2_act, self.v2, rtol=1e-3)
