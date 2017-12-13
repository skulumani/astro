"""Lamber solvers

Functions to solve Lambert's problem

Notes
-----
Much of this is based on my work in AAE532 at Purdue and Vallado.

Author
------
Shankar Kulumani		GWU		skulumani@gwu.edu
"""
import numpy as np

import logging

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
        h = np.norm(np.cross(r1,v1))
        
        p = h**2/mu
        
        ecc = np.sqrt(1-p/a)
        
        rp = a*(1-ecc)
    
    # TODO raise assertion error if rp  < r body

    return rp


def lambert_universal(r1, r2, direction, num_rev, tof, mu, r_body):
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
    if ( np.abssolute( vara ) > tol ):
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

    return v1, v2, errorl
