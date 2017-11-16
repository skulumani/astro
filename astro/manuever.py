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
def rvfpa2orbit_el(mag_r,mag_v,fpa,mu):
    """Converts R,V, and FPA to orbital elements

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
