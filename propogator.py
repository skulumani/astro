"""Start of propogator module

Hold many useful functions for numerical integration of astrodynamics
"""
from astro import constants
import numpy as np

def accel_third(mu_j, r_i2j, r_q2j):
    """Third body perturbations

    Calculate relative acceleration of third body (j) on object of
    interest (i) and it's reference object (q)

    Inputs: 
    - mu_j : gravitational parameters of third body in km^3/sec^2
    - r_i2j : position vector from particle of interest to the third body in km
    - r_q2j : position vector from the central body to the third body in km

    Outpus: 
    - direct : direct acceleration in km/sec^2
    - indirect : indirect acceleration in km/sec^2
    - perturbing : sum of direct and indirection

    Dependencies: 
    - None

    Author: 
        - Shankar Kulumani 8 Sept 2012
            - list revisions
        - Shankar Kulumani 25 Sept 2017
            - into Python

    References
        - AAE532 Lesson 6 31 Aug 2012 
        - MAE3145 HW2
    """

    direct = mu_j*r_i2j/np.linalg.norm(r_i2j)**3
    indirect = -mu_j*r_q2j/np.linalg.norm(r_q2j)**3

    perturbing = direct + indirect
    return direct, indirect, perturbing

def accel_twobody(mass_int, mass_body, r_body2int, G=constants.G):
    """Two body dominant acceleration

    """

    accel = -G * ( mass_body + mass_int) / np.linalg.norm(r_body2int)**3 * r_body2int
    return accel
