"""Coordinate transformations

"""

import numpy as np
from astro import kepler, constants, geodetic, time
from kinematics import attitude

# TODO: Add EOP and a full transformation here - compare to SPICE
def dcm_eci2ecef(jd):
    """Rotation matrix to convert from ECI to ECEF

    Will gradually improve this function to include all of the Earth
    motion terms. For now, just use sidereal time to get the rotation matrix
    """

    gst, _ = time.gstlst(jd, 0)

    dcm = attitude.rot3(gst, 'r')

    return dcm


def dcm_ecef2eci(jd):
    """Rotation matrix to transform from ECEF to ECI
    """
    dcm = dcm_eci2ecef(jd).T
    return dcm

# TODO: Documentation and unit testing
def dcm_pqw2eci_vector(r, v, mu=constants.earth.mu):
    """Define rotation matrix transforming PQW to ECI given eci vectors

    """
    h_vec, _, ecc_vec = kepler.hne_vec(r, v, mu)
    p_hat = ecc_vec / np.linalg.norm(ecc_vec)
    w_hat = h_vec / np.linalg.norm(h_vec) 
    q_hat = np.cross(w_hat, p_hat)

    dcm = np.stack((p_hat, q_hat, w_hat), axis=1)

    return dcm

#TODO: Documentation and testing
def dcm_pqw2eci_coe(raan, inc, arg_p):
    """Define rotation matrix transforming PQW to ECI given orbital elements

    """
    dcm = attitude.rot3(raan).dot(
            attitude.rot1(inc)).dot(attitude.rot3(arg_p))

    dcm_pqw2eci = attitude.rot3(-raan, 'r').dot(attitude.rot1(-inc, 'r')
                                                ).dot(attitude.rot3(-arg_p, 'r'))
    return dcm

def dcm_eci2pqw_coe(raan, inc, arg_p):
    """Define rotation matrix transforming PQW to ECI given orbital elements

    """
    dcm = dcm_pqw2eci_coe(raan, inc, arg_p).T
    return dcm

def dcm_pqw2lvlh(nu):
    dcm = attitude.rot3(-nu)
    return dcm

def dcm_lvlh2pqw(nu):
    dcm = dcm_pqw2lvlh(nu).T
    return dcm

#TODO: Documentation and testing
def dcm_sez2ecef(latgd, lon, alt=0):
    dcm = dcm_ecef2sez(latgd, lon)
    return dcm.T

#TODO: Documentation and testing
def dcm_ecef2sez(latgd, lon, alt=0):
    # dcm = attitude.rot2(np.pi/2 - latgd, 'r').dot(attitude.rot3(lon, 'r'))
    dcm = attitude.rot2(latgd - np.pi/2).dot(attitude.rot3(-lon))
    return dcm

#TODO: Documentation and testing
def dcm_ned2ecef(latgd, lon, alt=0):
    dcm = dcm_ecef2ned(latgd, lon, alt)
    return dcm.T

#TODO: Documentation and testing
def dcm_ecef2ned(latgd, lon, alt=0):
    dcm = np.array([[-np.sin(latgd)*np.cos(lon), -np.sin(lon), -np.cos(latgd)*np.cos(lon)],
                    [-np.sin(latgd)*np.sin(lon), np.cos(lon), -np.cos(latgd)*np.sin(lon)],
                    [np.cos(latgd), 0, -np.sin(latgd)]]).T
    return dcm

#TODO: Documentation and testing
def dcm_ecef2enu(latgd, lon, alt=0):
    dcm = np.array([[-np.sin(lon), np.cos(lon), 0],
                    [-np.sin(latgd)*np.cos(lon), -np.sin(latgd)*np.sin(lon), np.cos(latgd)],
                    [np.cos(latgd)*np.cos(lon), np.cos(latgd)*np.sin(lon), np.sin(latgd)]])

    return dcm

#TODO: Documentation and testing
def dcm_enu2ecef(latgd, lon, alt=0):
    dcm = dcm_ecef2enu(latgd, lon, alt)
    return dcm.T

# TODO: Add documentation and testing
# TODO: Make this vectorized to allow for vector inputs
def enu2ecefv(enu, latgd, lon, alt=0):

    # find rotation matrix for this location enu2ecef
    dcm_enu2ecef = dcm_enu2ecef(latgd, lon)

    # vector to site (latgd, lon)
    site_ecef = geodetic.lla2ecef(lat, lon, alt)

    # transform
    ecef = dcm_enu2ecef.dot(enu) + site_ecef

    return ecef

# TODO: Add documentation and testing
# TODO: Make this vectorized to allow for vector inputs
def ecef2enuv(ecef, latgd, lon, alt=0):
    dcm_ecef2enu = dcm_ecef2enu(latgd, lon)

    # site vector
    site_ecef = geodetic.lla2ecef(lat, lon, alt)

    # transform
    enu = dcm_ecef2enu.dot(ecef - site_ecef)
    return enu

# TODO: Add documentation and testing
# TODO: Make this vectorized to allow for vector inputs
def ned2ecefv(ned, latgd, lon, alt=0):

    # find rotation matrix for this location enu2ecef
    dcm_ned2ecef = dcm_ned2ecef(latgd, lon)

    # vector to site (latgd, lon)
    site_ecef = geodetic.lla2ecef(lat, lon, alt)

    # transform
    ecef = dcm_ned2ecef.dot(ned) + site_ecef

    return ecef

# TODO: Add documentation and testing
# TODO: Make this vectorized to allow for vector inputs
def ecef2nedv(ecef, latgd, lon, alt=0):
    dcm_ecef2ned = dcm_ecef2ned(latgd, lon)

    # site vector
    site_ecef = geodetic.lla2ecef(lat, lon, alt)

    # transform
    ned = dcm_ecef2ned.dot(ecef - site_ecef)
    return ned

# TODO: Add documentation and testing
# TODO: Make this vectorized to allow for vector inputs
def sez2ecefv(sez, latgd, lon, alt=0):

    # find rotation matrix for this location enu2ecef
    dcm_sez2ecef = dcm_sez2ecef(latgd, lon)

    # vector to site (latgd, lon)
    site_ecef = geodetic.lla2ecef(lat, lon, alt)

    # transform
    ecef = dcm_sez2ecef.dot(ned) + site_ecef

    return ecef

# TODO: Add documentation and testing
# TODO: Make this vectorized to allow for vector inputs
def ecef2sezv(ecef, latgd, lon, alt=0):
    dcm_ecef2sez = dcm_ecef2sez(latgd, lon)

    # site vector
    site_ecef = geodetic.lla2ecef(lat, lon, alt)

    # transform
    ned = dcm_ecef2sez.dot(ecef - site_ecef)
    return ned
