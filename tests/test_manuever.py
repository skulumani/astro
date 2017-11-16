import numpy as np
from astro import manuever, constants
import pdb
class TestRVFPA2OrbitEl():
    r = 4 * constants.earth.radius
    v = 4.54 # km /sec
    fpa = np.deg2rad(-40)
    a_true = 37477.21798
    p_true = 19751.017296
    ecc_true = 0.687739800
    nu_true = 4.377814485
    a, p, ecc, nu = manuever.rvfpa2orbit_el(r, v, fpa, constants.earth.mu)

    def test_semimajor_axis(self):
        """AAE532 PS6
        """
        np.testing.assert_allclose(self.a, self.a_true)

    def test_semiparameter(self):
        """AAE532 PS6
        """
        np.testing.assert_allclose(self.p, self.p_true)
    
    def test_eccentricy(self):
        np.testing.assert_allclose(self.ecc, self.ecc_true)

    def test_true_anomaly(self):
        np.testing.assert_allclose(self.nu, self.nu_true)




