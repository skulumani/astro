import numpy as np
from .. import constants


def test_newton_gravity_constant():
    G = 6.673e-20
    np.testing.assert_equal(constants.G, G)
