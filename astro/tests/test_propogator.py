"""Test out the propogator module
"""

import numpy as np
from astro import propogator, constants

def test_accel_third():
    """MAE3145 HW1 Problem 3"""
    mu = constants.sun.mu
    r_moon2sun = np.array([-constants.earth.orbit_sma, -constants.moon.orbit_sma])
    r_earth2sun = np.array([-constants.earth.orbit_sma, 0])
    direct, indirect, perturbation = propogator.accel_third(mu, r_moon2sun, r_earth2sun)
    np.testing.assert_allclose(direct, (-5.930645e-6, -1.523994e-8), rtol=1e-4)
    np.testing.assert_allclose(indirect, (5.930704e-6, 0), rtol=1e-4)
    np.testing.assert_allclose(perturbation, (5.874308e-11, -1.523994e-8), rtol=1e-4)

def test_accel_twobody():
    me = constants.earth.mass
    mb = constants.moon.mass
    r = np.array([0, constants.moon.orbit_sma])
    accel = propogator.accel_twobody(me, mb, r)
    np.testing.assert_allclose(accel, (0, -2.7307e-6), rtol=1e-4)

