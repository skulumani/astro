import numpy as np
# namedtuple to hold constants for each body
from astro import kepler, constants
import pdb

#TODO:Add test for tof_nu

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

class Testrv2coeVallado():

    p_true = 1.73527 * constants.earth.radius # km
    ecc_true = 0.83285
    inc_true = np.deg2rad(87.87)
    raan_true = np.deg2rad(227.89)
    arg_p_true = np.deg2rad(53.38)
    nu_true = np.deg2rad(92.335)
    mu = 398600.5 # km^3 /sec^2

    R_ijk_true = np.array([6524.834, 6862.875, 6448.296])
    V_ijk_true = np.array([4.901320, 5.533756, -1.976341]) 
    
    p, a, ecc, inc, raan, arg_p, nu, m, _, _, _= kepler.rv2coe(R_ijk_true, V_ijk_true, mu)
    
    rtol = 1e-3

    def test_p(self):
        np.testing.assert_allclose(self.p, self.p_true, rtol=self.rtol)
    
    def test_a(self):
        np.testing.assert_allclose(self.a, self.p_true / (1 - self.ecc_true**2), rtol=self.rtol)

    def test_ecc(self):
        np.testing.assert_allclose(self.ecc, self.ecc_true, rtol=self.rtol)
    
    def test_inc(self):
        np.testing.assert_allclose(self.inc, self.inc_true, rtol=self.rtol)

    def test_raan(self):
        np.testing.assert_allclose(self.raan, self.raan_true, rtol=self.rtol)

    def test_arg_p(self):
        np.testing.assert_allclose(self.arg_p, self.arg_p_true, rtol=self.rtol)
    
    def test_nu(self):
        np.testing.assert_allclose(self.nu, self.nu_true, rtol=self.rtol)

    def test_m(self):
        E_true, M_true = kepler.nu2anom(self.nu, self.ecc)
        np.testing.assert_allclose(self.m, M_true, rtol=self.rtol)


class Testrv2coeCurtis():
    """Using example 4.7 from Curtis
    """
    p_true = 80000**2/398600
    ecc_true = 1.4
    inc_true = np.deg2rad(30)
    raan_true = np.deg2rad(40)
    arg_p_true = np.deg2rad(60)
    nu_true = np.deg2rad(30)
    mu = 398600 # km^3 /sec^2

    R_ijk_true = np.array([-4040, 4815, 3629])
    V_ijk_true = np.array([-10.39, -4.772, 1.744]) 
    
    p, a, ecc, inc, raan, arg_p, nu, m, _, _, _= kepler.rv2coe(R_ijk_true, V_ijk_true, mu)
    
    rtol = 1e-3

    def test_p(self):
        np.testing.assert_allclose(self.p, self.p_true, rtol=self.rtol)
    
    def test_a(self):
        np.testing.assert_allclose(np.absolute(self.a), self.p_true / (self.ecc_true**2 - 1), rtol=1e2)

    def test_ecc(self):
        np.testing.assert_allclose(self.ecc, self.ecc_true, rtol=1e-1)
    
    def test_inc(self):
        np.testing.assert_allclose(self.inc, self.inc_true, rtol=self.rtol)

    def test_raan(self):
        np.testing.assert_allclose(self.raan, self.raan_true, rtol=self.rtol)

    def test_arg_p(self):
        np.testing.assert_allclose(self.arg_p, self.arg_p_true, rtol=self.rtol)
    
    def test_nu(self):
        np.testing.assert_allclose(self.nu, self.nu_true, rtol=self.rtol)

    def test_m(self):
        E_true, M_true = kepler.nu2anom(self.nu, self.ecc)
        np.testing.assert_allclose(self.m, M_true, rtol=self.rtol)

def test_coe2rv_curtis():
    """Test COE to RV for Curtis example 4.3"""

    p_true = 80000**2/398600
    ecc_true = 1.4
    inc_true = np.deg2rad(30)
    raan_true = np.deg2rad(40)
    arg_p_true = np.deg2rad(60)
    nu_true = np.deg2rad(30)
    mu = 398600 # km^3 /sec^2

    R_ijk_true = np.array([-4040, 4815, 3629])
    V_ijk_true = np.array([-10.39, -4.772, 1.744]) 
    
    R_ijk, V_ijk, R_pqw, V_pqw = kepler.coe2rv(p_true,ecc_true,inc_true,raan_true,arg_p_true,nu_true, mu)

    np.testing.assert_allclose(R_ijk,R_ijk_true, rtol=1e0)
    np.testing.assert_allclose(V_ijk,V_ijk_true, rtol=1e0)

def test_coe2rv_vallado():
    """Test COE to RV for Vallado example 2-6"""

    p_true = 1.735272 * constants.earth.radius # km
    ecc_true = 0.832853
    inc_true = np.deg2rad(87.87)
    raan_true = np.deg2rad(227.89)
    arg_p_true = np.deg2rad(53.38)
    nu_true = np.deg2rad(92.336)
    mu = 398600.5 # km^3 /sec^2

    R_ijk_true = np.array([6525.5454,6861.7313, 6449.0585])
    V_ijk_true = np.array([4.902227, 5.533085, -1.975757]) 
    
    R_ijk, V_ijk, R_pqw, V_pqw = kepler.coe2rv(p_true,ecc_true,inc_true,raan_true,arg_p_true,nu_true, mu)

    np.testing.assert_allclose(R_ijk,R_ijk_true, rtol=1e-4)
    np.testing.assert_allclose(V_ijk,V_ijk_true, rtol=1e-4)

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
    p = 8000
    ecc = 0
    mu = constants.earth.mu
    nu_0 = np.pi/2
    delta_t = 2 * np.pi * np.sqrt(p**3/ mu)
    
    # make sure you get back to the same spot
    E_f, M_f, nu_f = kepler.tof_delta_t( p, ecc, mu, nu_0, delta_t)
    np.testing.assert_allclose(nu_f, nu_0)

    

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
    mu = constants.earth.mu
    (state_eci, state_pqw, state_lvlh, state_sat_eci, state_sat_pqw,
     state_sat_lvlh) = kepler.conic_orbit(p, ecc, inc, raan, arg_p, nu_i, nu_f,
                                          mu)

    def test_x_axis(self):
        np.testing.assert_allclose(self.state_sat_eci[0], self.p)

    def test_y_axis(self):
        np.testing.assert_allclose(self.state_sat_eci[1], 0)

    def test_z_axis(self):
        np.testing.assert_allclose(self.state_sat_eci[2], 0)

    def test_circular_orbit(self):
        np.testing.assert_allclose(np.linalg.norm(self.state_eci[:, 0:3],
                                                  axis=1), self.p)
    
    def test_lvlh_radial_velocity(self):
        np.testing.assert_allclose(self.state_lvlh[:, 3], 0)

    def test_lvlh_tangential_velocity(self):
        np.testing.assert_allclose(self.state_lvlh[:, 4], np.sqrt(self.mu/self.p))

class TestConicOrbitHyperbolic():

    p = 10000 # km
    ecc = 2.0
    inc = 0.0
    raan = 0.0
    arg_p = 0.0
    nu_i = 0.0
    nu_f = 0.0 

    (state_eci, state_pqw, state_lvlh, state_sat_eci, state_sat_pqw,
     state_sat_lvlh) = kepler.conic_orbit(p, ecc, inc, raan, arg_p, nu_i, nu_f)
    
    def test_x_axis(self):
        np.testing.assert_allclose(self.state_sat_eci[0], self.p/(1-self.ecc**2)*(1-self.ecc))

    def test_y_axis(self):
        np.testing.assert_allclose(self.state_sat_eci[1], 0)

    def test_z_axis(self):
        np.testing.assert_allclose(self.state_sat_eci[2], 0)

class TestConicOrbitParabolic():

    p = 10000 # km
    ecc = 1.0
    inc = 0.0
    raan = 0.0
    arg_p = 0.0
    nu_i = 0.0
    nu_f = 0.0 

    (state_eci, state_pqw, state_lvlh, state_sat_eci, state_sat_pqw,
     state_sat_lvlh) = kepler.conic_orbit(p, ecc, inc, raan, arg_p, nu_i, nu_f)
    
    def test_x_axis(self):
        np.testing.assert_allclose(self.state_sat_eci[0], self.p/2)

    def test_y_axis(self):
        np.testing.assert_allclose(self.state_sat_eci[1], 0)

    def test_z_axis(self):
        np.testing.assert_allclose(self.state_sat_eci[2], 0)

class TestEllipticalOribtProperties():
    # test case from RV2COE Astro 321, MAE3145
    r = np.array([8840.0, 646, 5455])
    v = np.array([-0.695, 5.25, -1.65])

    r_mag_true = 10407.6866
    v_mag_true = 5.5469

    r_per_true = 6260.5311
    r_apo_true = 11134.4744
    energy_true = -22.9147
    period_true = 2.2423
    sma_true = 8697.5027
    ecc_true = 0.2802
    inc_true = 33.9987
    raan_true = 250.0287
    arg_p_true = 255.5372
    nu_true = 214.8548
    
    mu = constants.earth.mu

    p, a, ecc, inc, raan, arg_p, nu, _, _, _, _ = kepler.rv2coe(r, v, mu) 
    ( a, h, period, sme, fpa, r_per, r_apo, r_ijk, v_ijk,
     r_pqw, v_pqw, r_lvlh, v_lvlh, r, v, v_circ, v_esc,
     E, M, n ) = kepler.elp_orbit_el(p,ecc,inc,raan,arg_p,nu,mu)
    def test_r_mag(self):
        np.testing.assert_allclose(np.linalg.norm(self.r_ijk), self.r_mag_true, rtol=1e-4)

    def test_v_mag(self):
        np.testing.assert_allclose(np.linalg.norm(self.v_ijk), self.v_mag_true, rtol=1e-4)

    def test_r_per(self):
        np.testing.assert_allclose(self.r_per, self.r_per_true, rtol=1e-4)

    def test_r_apo(self):
        np.testing.assert_allclose(self.r_apo, self.r_apo_true, rtol=1e-4)

    def test_energy(self):
        np.testing.assert_allclose(self.sme, self.energy_true, rtol=1e-4)

    def test_period(self):
        np.testing.assert_allclose(self.period*constants.sec2hr, self.period_true, rtol=1e-4)

    def test_sma(self):
        np.testing.assert_allclose(self.a, self.sma_true, rtol=1e-4)

    def test_ecc(self):
        np.testing.assert_allclose(self.ecc, self.ecc_true, rtol=1e-4)

    def test_inc(self):
        np.testing.assert_allclose(self.inc*constants.rad2deg, self.inc_true, rtol=1e-4)

    def test_raan(self):
        np.testing.assert_allclose(self.raan*constants.rad2deg, self.raan_true, rtol=1e-4)

    def test_arg_p(self):
        np.testing.assert_allclose(self.arg_p*constants.rad2deg, self.arg_p_true, rtol=1e-4)

class TestHyperbolicOrbitProperties():
    """From MAE3145 HW4 Problem 2
    """
    mu = constants.earth.mu
    rp = 1000 + constants.earth.radius
    ecc = 1.05

    a_true = 147562.7399
    p_true = 15125.19
    energy_true = 1.350613644
    vinf_true = 1.64354108
    nu_inf_true = np.deg2rad(162.2472)
    flyby_true = np.deg2rad(144.4944196)

    a, p = kepler.hyp_per2sma(rp, ecc) 
    (a, v_inf, b, sme, flyby, nu_inf, h, fpa, r_per, r_ijk, v_ijk, r_pqw,
     v_pqw, r_lvlh, v_lvlh, r, v, v_circ, v_esc, H, M_H, n) = kepler.hyp_orbit_el(p, ecc, 0, 0, 0, np.pi/2, mu)
   
    def test_semi_major_axis(self):
        np.testing.assert_allclose(self.a, self.a_true)
    
    def test_specific_mechanical_energy(self):
        np.testing.assert_allclose(self.sme, self.energy_true)
    
    def test_velocity_infinity(self):
        np.testing.assert_allclose(self.v_inf, self.vinf_true)
    
    def test_true_anomaly_infinity(self):
        np.testing.assert_allclose(self.nu_inf, self.nu_inf_true)
    
    def test_flyby_angle(self):
        np.testing.assert_allclose(self.flyby, self.flyby_true)

class TestHNEVector_elliptical_equatorial():
    r, v, _, _ = kepler.coe2rv(10000, 0.2, 0, 0, 0, 0, constants.earth.mu)
    h, n, e = kepler.hne_vec(r, v, constants.earth.mu)

    def test_h_vec(self):
        np.testing.assert_allclose(self.h, np.array([0, 0, 1]))

    def test_n_vec(self):
        np.testing.assert_allclose(self.n, np.zeros(3))

    def test_e_vec(self):
        np.testing.assert_allclose(self.e, np.array([1, 0, 0]))
        
def test_perapo2aecc_circular():
    r_per = 8000
    r_apo = r_per
    a_actual = r_per
    p_actual = r_per
    ecc_actual = 0
    a, p, ecc = kepler.perapo2aecc(r_per, r_apo)
    np.testing.assert_allclose((a, p, ecc), (a_actual, p_actual, ecc_actual))

def test_perapo2aecc_semi_major_axis_mean():
    r_per = 10000
    r_apo = 20000
    a_actual = (r_per + r_apo) / 2
    a, _, _ = kepler.perapo2aecc(r_per, r_apo)
    np.testing.assert_allclose(a, a_actual)

class TestSemiLatusRectum():

    def test_circular(self):
        a = 8000
        ecc = 0
        p = kepler.semilatus_rectum(a, ecc)
        np.testing.assert_allclose(p, a)
    
    def test_elliptical(self):
        a = 8000
        ecc = 0.5
        p = kepler.semilatus_rectum(a, ecc)
        np.testing.assert_allclose(p, a * (1 - ecc**2))

    def test_parabolic(self):
        """Shouldn't work since parabolas have infinite a
        """
        a = np.infty
        ecc = 1
        p = kepler.semilatus_rectum(a, ecc)
        np.testing.assert_allclose(p, 0)
    
    def test_hyperbolic(self):
        a = -8000
        ecc = 1.5
        p = kepler.semilatus_rectum(a, ecc)
        np.testing.assert_allclose(p, a * (1 - ecc**2))

def test_hyperbolic_semi_major_axis():
    a_actual = -147562.739
    p_actual = 15125.180

    a, p = kepler.hyp_per2sma(1000 + 6378.137, 1.05)
    np.testing.assert_allclose(a, a_actual)
    np.testing.assert_allclose(p, p_actual)

def test_true_anomaly_solve():
    a_1 =6*6378.137
    ecc_1 = .5
    nu_0 = 0
    p_1 = a_1*(1-ecc_1**2)
    nu = kepler.nu_solve(p_1, ecc_1, 7.6 * 6378.137)
    np.testing.assert_allclose(nu[0], np.deg2rad(144.6654977))
