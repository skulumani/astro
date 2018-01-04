"""Lamber solvers

Functions to solve Lambert's problem

Notes
-----
Much of this is based on my work in AAE532 at Purdue and Vallado.

Author
------
Shankar Kulumani		GWU		skulumani@gwu.edu
"""
from kinematics import attitude
from astro import kepler

import numpy as np

import logging

# TODO Add unit test
def crash_check(r1, v1, r2, v2, mu, r_body):
    """Check if orbit intersects the surface

    rp = crash_check(r1, v1, r2, v2, mu, r_body)

    Inputs:
        - List all inputs into function

    Outputs:
        - List/describe outputs of function

    Author:
        - Shankar Kulumani 14 Nov 2012
        - Shankar Kulumani 16 Nov 2012
            - removed dot product check for ascending/descending segments.
            Now will always check for collision since long TOF transfer
            would not be caught. Rather waste some time than have a false
            negative.
        - Shankar Kulumani 13 Dec 2017
            - move to python

    References
        - AAE532
    """

    sme = np.linalg.norm(v1)**2/2 - mu/np.linalg.norm(r1)

    if sme > 0:
        a = mu/(2*sme)
        h = np.linalg.norm(np.cross(r1,v1))
        
        p = h**2/mu
        
        ecc = np.sqrt(p/a+1)
        
        rp = a*(ecc-1)
    elif sme < 0:
        a = -mu/(2*sme)
        h = np.linalg.norm(np.cross(r1,v1))
        
        p = h**2/mu
        
        ecc = np.sqrt(1-p/a)
        
        rp = a*(1-ecc)
    
    # TODO raise assertion error if rp  < r body

    return rp

def universal(r1, r2, direction, num_rev, tof, mu, r_body):
    """Lambert solver using universal variables

    ( v1,v2,errorl ) = lambert_universal ( r1,r2, direction, num_rev, tof,mu)

    Purpose: 
        - Solves lambert's problem using universal variable formulation and
        a bissection technique

    Inputs: 
        - r1 - initial position vector in km (1x3 or 3x1)
        - r2 - final position vector in km (1x3 or 3x1)
        - direction - direction of flight 'long' or 'short'
        - num_rev - number of revolutions (integer 0 to inf)
        - tof - flight time in seconds
        - mu - gravitational parameter of central body in km^3/sec^2
        - r_body - radius of attracting body in km

    Outputs: 
        - v1 - initial velocity vector in km/sec (same size as inputs)
        - v2 - final velocity vector in km/sec

    Dependencies: 
        - findc2c3.m - universal variable parameters
        - crash_check.m - check for crash into central body

    Author: 
        - Shankar Kulumani 7 Nov 2012
        - Shankar Kulumani 14 Nov 2012
            - added crash check
        - Shankar Kulumani 13 Dec 2017 
            - in python

    References
        - Vallado
        - AAE532 Notes LSN 30-34
    """
    tol = 1e-5 # can affect cases where znew is multiples of 2pi^2
    max_iter= 50

    magr1 = np.linalg.norm(r1)
    magr2  = np.linalg.norm(r2)

    v1 = np.zeros_like(r1)
    v2 = np.zeros_like(r2)

    cos_deltanu= num_rev + np.dot(r1,r2)/(magr1*magr2)

    if direction == 'long':
        vara = -np.sqrt( magr1*magr2*(1+cos_deltanu) )
    elif direction == 'short':
        vara =  np.sqrt( magr1*magr2*(1.0+cos_deltanu) )
    else:
        # TODO Throw exception and logging
        pass

    # initial guess
    psiold = 0.0
    psinew = 0.0
    xold   = 0.0
    c2new  = 0.5
    c3new  = 1/6

    # initial bounds for bissection
    if ( num_rev == 0 ):
        upper=  4*np.pi**2
        lower= -4*2*np.pi*np.pi
        num_rev = 0
    else:
        if num_rev == 1:
            upper=   16.0*np.pi*np.pi
            lower=    4.0*np.pi*np.pi
        else:
            upper=  36.0*np.pi*np.pi
            lower=  16.0*np.pi*np.pi

    #        chord = sqrt( magro^2 + magr^2 - 2.0*magro*magr*cosdeltanu );
    #            nrev = 1;
    #        chord = sqrt( magro^2 + magr^2 - 2.0*magro*magr*cosdeltanu );
    #        s     = ( magro + magr + chord )*0.5;
    #        betam = 2.0* asin( sqrt((s-chord)/chord) );  % comes out imaginary?? jst for hyperbolic??
    #        tmin  = ((2.0*nrev+1.0)*pi-betam + sin(betam))/sqrt(mu);

    # -------  determine if  the orbit is possible at all ---------
    if ( np.absolute( vara ) > tol ):
        loops  = 0
        ynegktr= 1  # y neg ktr
        dtnew = -10.0
        while ((np.absolute(dtnew-tof) >= tol) and (loops < max_iter) and (ynegktr <= 10)):
            
            if ( np.absolute(c2new) > tol ):
                y= magr1 + magr2 - ( vara*(1.0-psiold*c3new)/np.sqrt(c2new) )
            else:
                y= magr1 + magr2

            # ----------- check for negative values of y ----------
            if (  ( vara > 0.0 ) and ( y < 0.0 ) ):
                ynegktr= 1
                while (( y < 0.0 ) and ( ynegktr < 10 )):
                    psinew= 0.8*(1.0/c3new)*( 1.0 - (magr1+magr2)*np.sqrt(c2new)/vara  )

                    # -------- find c2 and c3 functions -----------
                    ( c2new,c3new ) = findc2c3( psinew )
                    psiold = psinew
                    lower  = psiold
                    if ( np.absolute(c2new) > tol ):
                        y= magr1 + magr2 - ( vara*(1.0-psiold*c3new)/np.sqrt(c2new) )
                    else:
                        y= magr1 + magr2
                    
                    ynegktr = ynegktr + 1
            
            if ( ynegktr < 10 ):
                if ( np.absolute(c2new) > tol ):
                    xold= np.sqrt( y/c2new )
                else:
                    xold= 0.0

                xoldcubed= xold**3
                dtnew    = (xoldcubed*c3new + vara*np.sqrt(y))/np.sqrt(mu)
                
                # --------  readjust upper and lower bounds -------
                if ( dtnew < tof ):
                    lower= psiold

                if ( dtnew > tof ):
                    upper= psiold

                psinew= (upper+lower) * 0.5
                
                # ------------- find c2 and c3 functions ----------
                ( c2new,c3new ) = findc2c3( psinew )
                psiold = psinew
                loops = loops + 1
                
                # --- make sure the first guess isn't too close ---
                if ( (np.absolute(dtnew - tof) < tol) and (loops == 1) ):
                    dtnew= tof-1.0
        
        if ( (loops >= max_iter) or (ynegktr >= 10) ):
            # TODO Throw assertion about not converging and log
            if ( ynegktr >= 10 ):
                # TODO Throw log that Y is negative
                print('\nERROR: Y Negative\n')
        else:
            # --- use f and g series to find velocity vectors -----
            f   = 1.0 - y/magr1
            gdot= 1.0 - y/magr2
            g   = vara*np.sqrt( y/mu )  # 1 over g
            
            v1 = (r2-f*r1)/g
            v2 = (gdot*r2-r1)/g
            
    else:
        # TODO Throw error about 180 deg transfer
        print('\nERROR: 180 deg transfer\n')

    crash_check(r1,v1,r2,v2,mu,r_body)

    return v1, v2

# TODO Add unit test
def findc2c3(psi):
    """Find C2 and C3 functions from psi

    [c2,c3] = findc2c3 ( psi )

    Inputs: 
        - psi - psi variable in rad

    Outputs: 
        - c2 - C2 function
        - c3 - C3 function

    Dependencies: 
        - none

    Author: 
        - Shankar Kulumani 16 Sept 2012
            - list revisions
        - Shankar Kulumani 13 Dec 2017
            - in python

    References
        - Vallado 3rd Edition Alg 1 pg 71  
    """


    small =     1e-6

    if ( psi > small ):
        sqrt_psi = np.sqrt( psi )
        c2 = (1.0 - np.cos( sqrt_psi )) / psi
        c3 = (sqrt_psi-np.sin( sqrt_psi )) / ( sqrt_psi**3 )
    else:
        if ( psi < -small ):
            sqrt_psi = np.sqrt( -psi )
            c2 = (1.0 -np.cosh( sqrt_psi )) / psi
            c3 = (np.sinh( sqrt_psi ) - sqrt_psi) / ( sqrt_psi**3 )
        else:
            c2 = 0.5
            c3 = 1/6

    return c2, c3

def minenergy(r1, r2, r_body, mu_body, direction):
    """Lambert minimum energy ellipse

   ( v1 v2 a p ecc ) = lambert_minenergy(r1,r2,r_body,mu_body)

   Purpose: 
       - Finds the minimum energy solution to lamberts problems given two
       position vectors.

   Inputs: 
       - r1 - initial position vector in km (1x3 or 3x1)
       - r2 - final position vector in km (1x3 or 3x1)
       - r_body - radius of central body in km (check for collision)
       - mu_body - gravitational parameter of central body in km^3/sec^2
       - direction - 'long' or 'short' direction of travel 
   Outputs: 
       - v1 - initial velocity vector in km/sec
       - v2 - final velocity vector in km/sec
       - a - semimajor axis of transfer ellipse in km
       - p - semiparameter of transfer ellipse in km
       - ecc - eccentricity of transfer ellipse
       - F - position vector of vacant focus in km 
       
   Dependencies: 
       - fg_nu - calculates velocity vectors using f and g functions
       - crash_check - check for crash into central body

   Author: 
       - Shankar Kulumani 5 Nov 2012
       - Shankar Kulumani 6 Nov 2012
           - added vacant focus calculation
       - Shankar Kulumani 14 Nov 2012
           - added crash check
        - Shankar Kulumani 13 Dec 2017
            - now in python

   References
       - AAE532 Notes LSN 30-33
       - Vallado 3rd edition
    """
    # find transfer angle
    mag_r1 = np.linalg.norm(r1)
    mag_r2 = np.linalg.norm(r2)
    cos_nu = np.dot(r1,r2)/(mag_r1*mag_r2)

    if direction == 'short':
        sin_nu = np.linalg.norm(np.cross(r1,r2))/(mag_r1*mag_r2)
    elif direction == 'long':
        sin_nu = -np.linalg.norm(np.cross(r1,r2))/(mag_r1*mag_r2)

    delta_nu = np.arctan2(sin_nu,cos_nu)
    # TODO Check on angle range for delta nu
    delta_nu = attitude.normalize(delta_nu, 0, 2* np.pi)

    # Intermediate parameters

    # c = sqrt(mag_r1^2+mag_r2^2-2*mag_r1*mag_r2*cos(delta_nu)); % Chord
    c_vec = r2-r1
    c = np.linalg.norm(c_vec)
    s = (mag_r1+mag_r2+c)/2 # Semiperimeter
    a = s/2
    # p = mag_r1*mag_r2/c*(1-cos(delta_nu)); % Semiparameter?

    alpha = 2*np.arcsin(np.sqrt(s/(2*a)))
    beta = 2*np.arcsin(np.sqrt((s-c)/(2*a)))

    p = 4*a*(s-mag_r1)*(s-mag_r2)/c**2*(np.sin(1/2*(alpha + beta))**2)

    # Transfer orbit physical properties

    ecc = np.sqrt(1-2*p/s)
    # ecc = c/(2*a);

    # vacant focus
    F = (2*a-mag_r1)*c_vec/c + r1
    # F = (2*a_m-norm(r1))*-c/norm(c) + r1

    # find velocity at each point
    # V0 = (sqrt(mu*p)/(r0*r*sin(v)))*(R - (1 - r/p*(1-cos(v)))*R0);
    ( v1, v2, f, g, f_dot, g_dot ) = kepler.fg_velocity(r1,r2,delta_nu,p,mu_body)

    # Transfer orbit time
    B = 2*np.arcsin(np.sqrt((s-c)/s))
    if delta_nu > np.pi:
        B = -B
    tof = np.sqrt(a**3/mu_body)*(np.pi - B + np.sin(B))

    # Check for crash
    crash_check(r1,v1,r2,v2,mu_body,r_body)

    return v1, v2, tof, a, p, ecc
