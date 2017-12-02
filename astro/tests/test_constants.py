import numpy as np
from astro import constants


def test_newton_gravity_constant():
    G = 6.673e-20
    np.testing.assert_equal(constants.G, G)

def test_rad2deg():
    np.testing.assert_allclose(np.pi/2*constants.rad2deg, np.rad2deg(np.pi/2))

def test_deg2rad():
    np.testing.assert_allclose(100*constants.deg2rad, np.deg2rad(100))

def test_sec2hr():
    np.testing.assert_allclose(3600*constants.sec2hr, 1)
    np.testing.assert_allclose(1*constants.hr2sec, 3600)

def test_sec2day():
    np.testing.assert_allclose(1 * constants.sec2day , 1 / (60 * 60 * 24))
    np.testing.assert_allclose(1 * constants.day2sec, 24 * 60 * 60)

