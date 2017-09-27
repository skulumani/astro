"""Test the transform module
"""

from .. import transform
import numpy as np

class TestPQWtoECI():
    raan = np.deg2rad(227.89)
    inc = np.deg2rad(87.87)
    argp = np.deg2rad(53.38)

    r = np.array([6524.834, 6862.875, 6448.296])
    v = np.array([4.901320, 5.533756, -1.976341])

    def test_both_functions_equivalent(self):
        dcm_pqw2eci_vector = transform.dcm_pqw2eci_vector(self.r, self.v)
        dcm_pqw2eci_coe = transform.dcm_pqw2eci_coe(self.raan, self.inc, self.argp)
        np.testing.assert_allclose(dcm_pqw2eci_coe, dcm_pqw2eci_vector, rtol=1e-3)
