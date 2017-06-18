import numpy as np
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

# define an Earth GEO stationary orbit

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

    
