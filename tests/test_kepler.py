import numpy as np
# namedtuple to hold constants for each body
from .. import kepler

def test_coe2rv_equatorial_circular():
    """Test COE to RV for equatorial circular orbit around Earth"""

    p = 6378.137 # km
    ecc = 0.0
    inc = 0.0
    raan = 0.0
    arg_p = 0.0
    nu = 0.0
    mu = 398600.5 # km^3 /sec^2

    R_ijk_true = np.array([6378.137,0,0])
    V_ijk_true = np.sqrt(mu/p) * np.array([0,1,0])
    R_pqw_true = R_ijk_true
    V_pqw_true = V_ijk_true

    R_ijk, V_ijk, R_pqw, V_pqw = kepler.coe2rv(p,ecc,inc,raan,arg_p,nu, mu)

    np.testing.assert_array_almost_equal(R_ijk_true,R_ijk)
    np.testing.assert_array_almost_equal(V_ijk_true,V_ijk)
    np.testing.assert_array_almost_equal(V_pqw_true,V_pqw)
    np.testing.assert_array_almost_equal(V_pqw_true,V_pqw)

class Testrv2coeEquatorialCircular():

    p_true = 6378.137 # km
    ecc_true = 0.0
    inc_true = 0.0
    raan_true = 0.0
    arg_p_true = 0.0
    nu_true = 0.0
    mu = 398600.5 # km^3 /sec^2

    R_ijk_true = np.array([6378.137,0,0])
    V_ijk_true = np.sqrt(mu/p_true) * np.array([0,1,0])
    
    p, a, ecc, inc, raan, arg_p, nu, m, arglat, truelon, lonper = kepler.rv2coe(R_ijk_true, V_ijk_true, mu)
    
    def test_p(self):
        np.testing.assert_allclose(self.p, self.p_true)
    
    def test_a(self):
        np.testing.assert_allclose(self.a, self.p_true / (1 - self.ecc_true**2))

    def test_ecc(self):
        np.testing.assert_allclose(self.ecc, self.ecc_true)
    
    def test_inc(self):
        np.testing.assert_allclose(self.inc, self.inc_true)

    def test_raan(self):
        np.testing.assert_allclose(self.raan, self.raan_true)

    def test_arg_p(self):
        np.testing.assert_allclose(self.arg_p, self.arg_p_true)
    
    def test_nu(self):
        np.testing.assert_allclose(self.nu, self.nu_true)

    def test_m(self):
        E_true, M_true = kepler.nu2anom(self.nu, self.ecc)
        np.testing.assert_allclose(self.m, M_true)

    def test_arglat(self):
        np.testing.assert_allclose(self.arglat, self.nu_true + self.arg_p_true)

    def test_truelon(self):
        np.testing.assert_allclose(self.truelon, self.nu_true + self.raan_true + self.arg_p_true)

    def test_lonper(self):
        np.testing.assert_allclose(self.lonper, self.arg_p_true + self.raan_true)

def test_coe2rv_polar_circular():
    """Test COE to RV for polar circular orbit around Earth"""

    p = 6378.137 # km
    ecc = 0.0
    inc = np.pi/2
    raan = 0.0
    arg_p = 0.0
    nu = 0.0
    mu = 398600.5 # km^3 /sec^2

    R_ijk_true = np.array([6378.137,0,0])
    V_ijk_true = np.sqrt(mu/p) * np.array([0.0,0.0,1])

    R_ijk, V_ijk, R_pqw, V_pqw = kepler.coe2rv(p,ecc,inc,raan,arg_p,nu, mu)

    np.testing.assert_array_almost_equal(R_ijk_true,R_ijk)
    np.testing.assert_array_almost_equal(V_ijk_true,V_ijk)

class Testrv2coePolarCircular():

    p_true = 6378.137 # km
    ecc_true = 0.0
    inc_true = np.pi / 2
    raan_true = 0.0
    arg_p_true = 0.0
    nu_true = 0.0
    mu = 398600.5 # km^3 /sec^2

    R_ijk_true = np.array([6378.137,0,0])
    V_ijk_true = np.sqrt(mu/p_true) * np.array([0,0,1])
    
    p, a, ecc, inc, raan, arg_p, nu, m, arglat, truelon, lonper = kepler.rv2coe(R_ijk_true, V_ijk_true, mu)
    
    def test_p(self):
        np.testing.assert_allclose(self.p, self.p_true)
    
    def test_a(self):
        np.testing.assert_allclose(self.a, self.p_true / (1 - self.ecc_true**2))

    def test_ecc(self):
        np.testing.assert_allclose(self.ecc, self.ecc_true)
    
    def test_inc(self):
        np.testing.assert_allclose(self.inc, self.inc_true)

    def test_raan(self):
        np.testing.assert_allclose(self.raan, self.raan_true)

    def test_arg_p(self):
        np.testing.assert_allclose(self.arg_p, self.arg_p_true)
    
    def test_nu(self):
        np.testing.assert_allclose(self.nu, self.nu_true)

    def test_m(self):
        E_true, M_true = kepler.nu2anom(self.nu, self.ecc)
        np.testing.assert_allclose(self.m, M_true)

    def test_arglat(self):
        np.testing.assert_allclose(self.arglat, self.nu_true + self.arg_p_true)

    def test_truelon(self):
        np.testing.assert_allclose(self.truelon, self.nu_true + self.raan_true + self.arg_p_true)

    def test_lonper(self):
        np.testing.assert_allclose(self.lonper, self.arg_p_true + self.raan_true)

def test_coe2rv_equatorial_circular_quarter():
    """Test COE to RV for equatorial circular orbit around Earth"""

    p = 6378.137 # km
    ecc = 0.0
    inc = 0.0
    raan = 0.0
    arg_p = 0.0
    nu = np.pi/2
    mu = 398600.5 # km^3 /sec^2

    R_ijk_true = np.array([0,6378.137,0])
    V_ijk_true = np.sqrt(mu/p) * np.array([-1.0,0.0,0])

    R_ijk, V_ijk, R_pqw, V_pqw = kepler.coe2rv(p,ecc,inc,raan,arg_p,nu, mu)

    np.testing.assert_array_almost_equal(R_ijk_true,R_ijk)
    np.testing.assert_array_almost_equal(V_ijk_true,V_ijk)    

class Testrv2coeEquatorialCircularQuarer():

    p_true = 6378.137 # km
    ecc_true = 0.0
    inc_true = 0.0
    raan_true = 0.0
    arg_p_true = 0.0
    nu_true = np.pi /2
    mu = 398600.5 # km^3 /sec^2

    R_ijk_true = np.array([0.0, 6378.137,0])
    V_ijk_true = np.sqrt(mu/p_true) * np.array([-1,0,0])
    
    p, a, ecc, inc, raan, arg_p, nu, m, arglat, truelon, lonper = kepler.rv2coe(R_ijk_true, V_ijk_true, mu)
    
    def test_p(self):
        np.testing.assert_allclose(self.p, self.p_true)
    
    def test_a(self):
        np.testing.assert_allclose(self.a, self.p_true / (1 - self.ecc_true**2))

    def test_ecc(self):
        np.testing.assert_allclose(self.ecc, self.ecc_true)
    
    def test_inc(self):
        np.testing.assert_allclose(self.inc, self.inc_true)

    def test_raan(self):
        np.testing.assert_allclose(self.raan, self.raan_true)

    def test_arg_p(self):
        np.testing.assert_allclose(self.arg_p, self.arg_p_true)
    
    def test_nu(self):
        np.testing.assert_allclose(self.nu, self.nu_true)

    def test_m(self):
        E_true, M_true = kepler.nu2anom(self.nu, self.ecc)
        np.testing.assert_allclose(self.m, M_true)

    def test_arglat(self):
        # argument of latitude isn't used for this case so it goes to zero
        # np.testing.assert_allclose(self.arglat, self.nu_true + self.arg_p_true)
        pass

    def test_truelon(self):
        np.testing.assert_allclose(self.truelon, self.nu_true + self.raan_true + self.arg_p_true)

    def test_lonper(self):
        np.testing.assert_allclose(self.lonper, self.arg_p_true + self.raan_true)

def test_coe2rv_equatorial_circular_half():
    """Test COE to RV for equatorial circular orbit around Earth"""

    p = 6378.137 # km
    ecc = 0.0
    inc = 0.0
    raan = 0.0
    arg_p = 0.0
    nu = np.pi
    mu = 398600.5 # km^3 /sec^2

    R_ijk_true = np.array([-p,0,0])
    V_ijk_true = np.sqrt(mu/p) * np.array([0.0,-1.0,0])

    R_ijk, V_ijk, R_pqw, V_pqw = kepler.coe2rv(p,ecc,inc,raan,arg_p,nu, mu)

    np.testing.assert_array_almost_equal(R_ijk_true,R_ijk)
    np.testing.assert_array_almost_equal(V_ijk_true,V_ijk)

class Testrv2coeRV1():
    a_true = 8697.5027
    ecc_true = 0.2802
    p_true = a_true * ( 1 - ecc_true**2)
    inc_true = np.deg2rad(33.9987)
    raan_true = np.deg2rad(250.0287)
    arg_p_true = np.deg2rad(255.5372)
    nu_true = np.deg2rad(214.8548)
    mu = 398600.5

    r = np.array([8840, 646, 5455])
    v = np.array([-0.6950, 5.250, -1.65])

    p, a, ecc, inc, raan, arg_p, nu, m, arglat, truelon, lonper = kepler.rv2coe(r, v, mu)

    def test_p(self):
        np.testing.assert_allclose(self.p, self.p_true, rtol=1e-1)
    
    def test_a(self):
        np.testing.assert_allclose(self.a, self.p_true / (1 - self.ecc_true**2))

    def test_ecc(self):
        np.testing.assert_allclose(self.ecc, self.ecc_true, rtol=1e-4)
    
    def test_inc(self):
        np.testing.assert_allclose(self.inc, self.inc_true, rtol=1e-4)

    def test_raan(self):
        np.testing.assert_allclose(self.raan, self.raan_true)

    def test_arg_p(self):
        np.testing.assert_allclose(self.arg_p, self.arg_p_true, rtol=1e-4)
    
    def test_nu(self):
        np.testing.assert_allclose(self.nu, self.nu_true, rtol=1e-4)

    def test_m(self):
        E_true, M_true = kepler.nu2anom(self.nu, self.ecc)
        np.testing.assert_allclose(self.m, M_true)

    def test_arglat(self):
        # argument of latitude isn't used for this case so it goes to zero
        # np.testing.assert_allclose(self.arglat, self.nu_true + self.arg_p_true)
        pass

    def test_truelon(self):
        # not used for inclined orbits - only special ones
        # np.testing.assert_allclose(self.truelon, self.nu_true + self.raan_true + self.arg_p_true)
        pass

    def test_lonper(self):
        # not really used for inclined orbits
        # np.testing.assert_allclose(self.lonper, self.arg_p_true + self.raan_true)
        pass

class Testrv2coeEquatorialCircularHalf():

    p_true = 6378.137 # km
    ecc_true = 0.0
    inc_true = 0.0
    raan_true = 0.0
    arg_p_true = 0.0
    nu_true = np.pi
    mu = 398600.5 # km^3 /sec^2

    R_ijk_true = np.array([-p_true, 0,0])
    V_ijk_true = np.sqrt(mu/p_true) * np.array([0,-1,0])
    
    p, a, ecc, inc, raan, arg_p, nu, m, arglat, truelon, lonper = kepler.rv2coe(R_ijk_true, V_ijk_true, mu)
    
    def test_p(self):
        np.testing.assert_allclose(self.p, self.p_true)
    
    def test_a(self):
        np.testing.assert_allclose(self.a, self.p_true / (1 - self.ecc_true**2))

    def test_ecc(self):
        np.testing.assert_allclose(self.ecc, self.ecc_true)
    
    def test_inc(self):
        np.testing.assert_allclose(self.inc, self.inc_true)

    def test_raan(self):
        np.testing.assert_allclose(self.raan, self.raan_true)

    def test_arg_p(self):
        np.testing.assert_allclose(self.arg_p, self.arg_p_true)
    
    def test_nu(self):
        np.testing.assert_allclose(self.nu, self.nu_true)

    def test_m(self):
        E_true, M_true = kepler.nu2anom(self.nu, self.ecc)
        np.testing.assert_allclose(self.m, M_true)

    def test_arglat(self):
        # argument of latitude isn't used for this case so it goes to zero
        # np.testing.assert_allclose(self.arglat, self.nu_true + self.arg_p_true)
        pass

    def test_truelon(self):
        np.testing.assert_allclose(self.truelon, self.nu_true + self.raan_true + self.arg_p_true)

    def test_lonper(self):
        np.testing.assert_allclose(self.lonper, self.arg_p_true + self.raan_true)
def test_kepler_eq_E():
    """
        A series of cases run in Matlab and copied here
    """
    M = np.deg2rad(110)
    ecc = 0.9
    E_matlab = 2.475786297687611 
    nu_matlab = 2.983273149717047

    E_python, nu_python, count_python = kepler.kepler_eq_E(M,ecc)

    np.testing.assert_allclose((E_python, nu_python),(E_matlab,nu_matlab))

def test_kepler_eq_E_zero():
    """
        Make sure that at zero nu=E=M
    """
    M = 0.0
    ecc = 1.2
    E_true = 0.0
    nu_true = 0.0
    
    E_python, nu_python, count_python = kepler.kepler_eq_E(M,ecc)

    np.testing.assert_array_almost_equal((E_python,nu_python),(E_true,nu_true))


def test_kepler_eq_E_pi():
    """
        Make sure that at pi nu=E=M
    """
    M = np.pi
    ecc = 0.9
    E_true = np.pi
    nu_true = np.pi

    E_python, nu_python, count_python = kepler.kepler_eq_E(M,ecc)

    np.testing.assert_array_almost_equal((E_python,nu_python),(E_true,nu_true))

def test_nu2anom():
    """
        A matlab test case which is copied here
    """
    M_matlab = np.deg2rad(110)
    ecc = 0.9
    E_matlab = 2.475786297687611 
    nu_matlab = 2.983273149717047

    E_python, M_python = kepler.nu2anom(nu_matlab,ecc)

    np.testing.assert_allclose(E_python, E_matlab)
    np.testing.assert_allclose(M_python, M_matlab)

def test_nu2anom_zero():
    """
        Make sure that at zero nu=E=M
    """
    M_true = 0.0
    ecc = 1.2
    E_true = 0.0
    nu_true = 0.0

    E_python, M_python = kepler.nu2anom(nu_true,ecc)

    np.testing.assert_allclose((E_python, M_python),(E_true,M_true))


def test_nu2anom_pi():
    """
        Make sure that at pi nu=E=M
    """
    M_true = np.pi
    ecc = 0.9
    E_true = np.pi
    nu_true = np.pi
    
    E_python, M_python = kepler.nu2anom(nu_true,ecc)

    np.testing.assert_allclose(E_python, E_true)
    np.testing.assert_allclose(M_python, M_true)


def test_tof_delta_t():
    """Test propogation using Kepler's Eq"""

    # define circular orbit around the Earth

    # find period of orbit and propogate over 1 period

    # make sure you get back to the same spot

    

def test_fpa_solve_circular():
    fpa_actual = kepler.fpa_solve(0, 0)
    fpa_expected = 0 
    np.testing.assert_allclose(fpa_actual, fpa_expected)

def test_fpa_solve_elliptical():
    fpa_actual = kepler.fpa_solve(0, 0.5)
    fpa_expected = 0
    np.testing.assert_allclose(fpa_actual, fpa_expected)

def test_fpa_solve_parabolic():
    fpa_actual = kepler.fpa_solve(0, 1)
    fpa_expected = 0
    np.testing.assert_allclose(fpa_actual, fpa_expected)

def test_fpa_solve_hyperbolic():
    fpa_actual = kepler.fpa_solve(0, 2)
    fpa_expected = 0
    np.testing.assert_allclose(fpa_actual, fpa_expected)

class TestConicOrbitEquatorial():

    p = 7000 # km
    ecc = 0.0
    inc = 0.0
    raan = 0.0
    arg_p = 0.0
    nu_i = 0.0
    nu_f = 0.0 

    x, y, z, xs, ys, zs = kepler.conic_orbit(p, ecc, inc, raan, arg_p, nu_i, nu_f)
    
    def test_x_axis(self):
        np.testing.assert_allclose(self.xs, self.p)

    def test_y_axis(self):
        np.testing.assert_allclose(self.ys, 0)

    def test_z_axis(self):
        np.testing.assert_allclose(self.zs, 0)

    def test_circular_orbit(self):
        orbit = np.stack((self.x, self.y, self.z), axis=1)
        np.testing.assert_allclose(np.linalg.norm(orbit, axis=1), self.p)

class TestConicOrbitHyperbolic():

    p = 10000 # km
    ecc = 2.0
    inc = 0.0
    raan = 0.0
    arg_p = 0.0
    nu_i = 0.0
    nu_f = 0.0 

    x, y, z, xs, ys, zs = kepler.conic_orbit(p, ecc, inc, raan, arg_p, nu_i, nu_f)
    
    def test_x_axis(self):
        np.testing.assert_allclose(self.xs, self.p/(1-self.ecc**2)*(1-self.ecc))

    def test_y_axis(self):
        np.testing.assert_allclose(self.ys, 0)

    def test_z_axis(self):
        np.testing.assert_allclose(self.zs, 0)

class TestConicOrbitParabolic():

    p = 10000 # km
    ecc = 1.0
    inc = 0.0
    raan = 0.0
    arg_p = 0.0
    nu_i = 0.0
    nu_f = 0.0 

    x, y, z, xs, ys, zs = kepler.conic_orbit(p, ecc, inc, raan, arg_p, nu_i, nu_f)
    
    def test_x_axis(self):
        np.testing.assert_allclose(self.xs, self.p/2)

    def test_y_axis(self):
        np.testing.assert_allclose(self.ys, 0)

    def test_z_axis(self):
        np.testing.assert_allclose(self.zs, 0)
