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
from scipy.optimize import fsolve
# TODO: Add documentation


def rvfpa2orbit_el(mag_r, mag_v, fpa, mu):
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
    a = -mu / (mag_v**2 / 2 - mu / mag_r) / 2
    sme = -mu / (2 * a)

    # find semi latus rectum and eccentricity
    mag_h = mag_r * mag_v * np.cos(fpa)
    p = mag_h**2 / mu

    ecc = np.sqrt(-p / a + 1)

    # find true anomaly using conic equation
    nu = np.arccos((p / mag_r - 1) / ecc)
    if fpa < 0:
        nu = - nu
    nu = attitude.normalize(nu, 0, 2 * np.pi)[0]
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
    beta = np.pi - alpha
    mag_vf = np.sqrt(mag_v1**2 + delta_v**2 - 2 *
                     mag_v1 * delta_v * np.cos(beta))

    delta_fpa = np.arccos(
        (delta_v**2 - mag_vf**2 - mag_v1**2) / (-2 * mag_vf * mag_v1))

    mag_rf = mag_r1

    fpaf = fpa1 + delta_fpa

    return mag_rf, mag_vf, fpaf


def nu_solve(r, v, fpa, mu):
    """Single impulse manuever true anomaly 
        nu = nu_solve(r,v,fpa)

     Inputs: 
        - r - magnitude of position vector in km
        - v - magnitude of velocity vector in km
        - fpa - flight path angle in rad -pi/2 < fpa < pi/2
        - mu - gravitational parameter km^3/sec^2

     Outputs: 
        - nu - true anomaly of body in rad 0 < nu < 2*pi

     Dependencies: 
        - none

     Author: 
        - Shankar Kulumani 10 Oct 2012
        - Shankar Kulumani 18 November 2017 
            - in Python

    References
        - AAE532 notes LSN 18
        - AAE532_PS7.pdf
    """
    num = (r * v**2 / mu) * np.cos(fpa) * np.sin(fpa)
    den = (r * v**2 / mu) * np.cos(fpa)**2 - 1

    nu = np.arctan2(num, den)

    nu = attitude.normalize(nu, 0, 2*np.pi)
    return nu

def delta_v_solve_planar(v1, v2, fpa1, fpa2):
    """Solve for DeltaV using law of cosines for planar manuevers
    Delta V Law of Cosines

    (delta_v, alpha, beta) = delta_v_solve(v_1,v_2,fpa_1,fpa_2)

    Inputs: 
        - v_1 - velocity magnitude of initial orbit in km/sec
        - v_2 - velocity magnitude of final orbit in km/sec
        - fpa_1 - flight path angle of initial orbit in rad
        - fpa_2 - flight path angle of final orbit in rad

    Outputs: 
        - delta_v - delta_v required using the law of cosines
        - alpha - exterior angle of delta_v wrt to v_1 using law of sines
        - beta - interior angle between v1 and delta_v
    Dependencies: 
        - none

    Author: 
        - Shankar Kulumani 10 Oct 2012
        - Shankar Kulumani 30 Oct 2012
            - modified beta angle calculation
        - Shankar Kulumani 18 Nov 2017
            - Move to Python

    References
        - AAE532 notes LSN 19
        - AAE532_PS7.pdf
        - MAE3145 HW5 2017
    """
    delta_fpa = np.absolute(fpa2-fpa1)
    delta_v = np.sqrt(v2**2+v1**2-2*v2*v1*np.cos(delta_fpa))

    # need logic here to solve sign ambiguity
    sin_beta = v2*np.sin(delta_fpa)/delta_v
    cos_beta = (v2**2 - v1**2 - delta_v**2) / (-2 * delta_v * v1)
    beta = np.arctan2(sin_beta, cos_beta)
    alpha = np.pi-beta
    return delta_v, alpha, beta

# TODO: Add documentation
def planar_conic_orbit_intersection(p1, p2, ecc1, ecc2, dargp, nu1_old=np.deg2rad(90)):
    """HW6 Problem 3"""
    # Find nu1 for the original orbit
    
    def f(nu1):
        f = p2/(1+ecc2*np.cos(nu1-dargp)) - (p1/(1+ecc1*np.cos(nu1)))
        return f

    def fdot(nu1):
        fdot = ((p2*ecc2*np.sin(nu1-dargp))/(1+ecc2*np.cos(nu1-dargp))**2)-((p1*ecc1*np.sin(nu1))/(1+ecc1*np.cos(nu1))**2)
        return fdot
    nu1_new = fsolve(f, nu1_old)    
    # count = 0
    # delta = 1
    # while delta > 1e-9 and count < 50:
    #     count = count+1
        
    #     nu1_new = nu1_old - f(nu1_old)/fdot(nu1_old)
        
    #     delta = np.absolute(nu1_new - nu1_old)

    return nu1_new[0]

def delta_v_vnc(dv_mag, alpha, beta, fpa):
    """VNC LVLH Delta V vector

    [dv_vnc dv_lvlh] = delta_v_vnc(dv_mag,alpha, beta, fpa)

    Purpose: 
        - Converts the alpha/beta angle of the Delta_V to the VNC frame
            - V - tangent to current velocity vector
            - N - normal to orbit plane (same direction as ang_mom vector)
            - C - orthongonal to velocity vector in orbit plane
        - Converts Delta_v into the LVLH frame
            - r_hat - in direction of orbit position vector
            - theta_hat - normal to r_hat in orbit plane
            - h_hat - orthogonal to orbit plane

    Inputs: 
        - dv_mag - magnitude of delta_v in km/sec
        - alpha - angle between projection of delta_v vector into the orbit
        plane and the orginal velocity vector (V) in rad
        - beta - angle of delta_v vector out of orbit plane (V-C plane) in
        rad
        - fpa - current flight path angle in rad - angle of the velocity
        vector wrt to theta_hat

    Outputs: 
        - dv_vnc - delta_v vector in VNC frame [V_hat;C_hat;N_hat];
        - dv_lvlh - delta_v vector in LVLH frame [r_hat;theta_hat;h_hat]

    Dependencies: 
        - none

    Author: 
        - Shankar Kulumani 15 Oct 2012
        - Shankar Kulumani 18 Nov 2017
            - In python

    References
        - AAE532 notes LSN 20
        - Vallado - referred to as NTW frame or Frenet System
    """

    dv_vnc = dv_mag*np.array([np.cos(beta)*np.cos(alpha), np.cos(beta)*np.sin(alpha), np.sin(beta)])

    phi = fpa+alpha

    dv_lvlh = dv_mag* np.array([np.cos(beta)*np.sin(phi),np.cos(beta)*np.cos(phi),np.sin(beta)])

    return dv_vnc, dv_lvlh
