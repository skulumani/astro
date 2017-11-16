"""Module for orbital manuevers

Extended description of the module

Notes
-----
    This is an example of an indented section. It's like any other section,
    but the body is indented to help it stand out from surrounding text.

If a section is indented, then a section break is created by
resuming unindented text.

Attributes
----------
module_level_variable1 : int
    Descrption of the variable

Author
------
Shankar Kulumani		GWU		skulumani@gwu.edu
"""

import numpy as np
from kinematics import attitude
import pdb
# TODO: Add documentation
def rvfpa2orbit_el(mag_r,mag_v,fpa,mu):
    """Converts R,V, and FPA to orbital elements
    
    a, p, ecc, nu = rvfpa2orbit_el(mag_r, mag_v, fpa, mu)

    Inputs: 
        - List all inputs into function

    Outputs: 
        - List/describe outputs of function

    Dependencies: 
        - list dependent m files required for function

    Author: 
        - Shankar Kulumani 7 Oct 2012
        - Shankar Kulumani 5 Nov 2012
            - added ascending/descending logic to true anomaly
        - Shankar Kulumani 16 November 2017
            - move to python
    References
        - AAE532_PS6.pdf
        """
    # calculate semi major axis using energy relationship
    a = -mu/(mag_v**2/2-mu/mag_r)/2
    sme = -mu/(2*a)

    # find semi latus rectum and eccentricity
    mag_h = mag_r*mag_v*np.cos(fpa)
    p = mag_h**2/mu

    ecc = np.sqrt(-p/a+1)

    # find true anomaly using conic equation
    nu = np.arccos((p/mag_r-1)/ecc)
    if fpa < 0:
        nu = - nu
    nu = attitude.normalize(nu, 0, 2*np.pi)[0]
    # orbit_el(p,ecc,0,0,0,nu,mu,'true')

    return a, p, ecc, nu

def single_impulse(mag_r1, mag_v1, fpa1, delta_v, alpha):
    """Given a deltaV and alpha angle outputs the new orbital elements

    mag_rf, mag_vf, fpaf = single_impulse(mag_r1, mag_v1, fpa1, delta_v, alpha)      

   Inputs: 
       - List all inputs into function

   Outputs: 
       - List/describe outputs of function

   Dependencies: 
       - list dependent m files required for function

   Author: 
       - Shankar Kulumani 7 Oct 2012
           - list revisions

   References
       - AAE532_PS6.pdf  
    """
    beta = np.pi-alpha
    mag_vf = np.sqrt(mag_v1**2+delta_v**2 -2*mag_v1*delta_v*np.cos(beta))

    delta_fpa = np.arccos((delta_v**2-mag_vf**2 - mag_v1**2)/(-2*mag_vf*mag_v1))

    mag_rf = mag_r1

    fpaf= fpa1+delta_fpa

    return mag_rf, mag_vf, fpaf
