"""Coordinate transformations

"""

import numpy as np
from astro import kepler, constants
from kinematics import attitude

def dcm_pqw2eci_vector(r, v, mu=constants.earth.mu):
    """Define rotation matrix transforming PQW to ECI given eci vectors

    """
    h_vec, _, ecc_vec = kepler.hne_vec(r, v, mu)
    p_hat = ecc_vec / np.linalg.norm(ecc_vec)
    w_hat = h_vec / np.linalg.norm(h_vec) 
    q_hat = np.cross(w_hat, p_hat)

    dcm = np.stack((p_hat, q_hat, w_hat), axis=1)

    return dcm

def dcm_pqw2eci_coe(raan, inc, arg_p):
    """Define rotation matrix transforming PQW to ECI given orbital elements

    """
    dcm = attitude.rot3(raan).dot(
            attitude.rot1(inc)).dot(attitude.rot3(arg_p))

    return dcm
