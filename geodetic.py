"""Geodetic transformations. This module holds many functions to peform common
transformations used in astrodynamics. 

Copyright (C) {2017}  {Shankar Kulumani}

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np
from kinematics import attitude


def lla2ecef(lat, lon, alt, r=6378.137, ee=8.1819190842622e-2):
    """Convert latitude, longitude, and altitude to the Earth centered
    Earth fixed frame (ECEF)

    Will convert between geodetic latitude, longitude, and altitude to the
    associated planet fixed reference frame.  This assumes an oblate spheroid
    model for the planetary body. It is applicable to any spherical body given
    the semi-major axis and the body eccentricity constant.

    Parameters:
    -----------
    lat : float
        Geodetic latitude (radians)
    lon : float
        Geodetic longitude (radians)
    alt : float
        height about the mean surface ellipsoid (WGS84 for Earth) (kilometers)

    Returns
    -------
    ecef : float ndarray (3,)
        Body fixed position vector in (kilometers)

    Notes
    -----
    We're assuming an oblate spheroid model for the Earth. To transform this
    vector to the inertial ECI frame all you need to do is rotate it by the
    Greenwich Mean Sideral time (GST) or use a fancy ECEF-ECI rotation which
    accounts for the Earth's precession/nutation.

    References
    ----------
    .. [1] BATE, Roger R, MUELLER, Donald D WHITE, Jerry E. Fundamentals of
    Astrodynamics. Courier Dover Publications, 1971.

    .. [2] Nice website to verify computations:
    http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm

    Examples
    --------
    Some examples demonstrating the usage of this function

    >>> import numpy

    Convert the lattitude, longitude, and altitude of Washington, DC to it's
    equivalent ECEF representation.

    >>> lat, lon, alt = (38.8895 * np.pi / 180, -77.035 * np.pi / 180, 0.054)

    We need to make sure the inputs are in the correct units.

    >>> lla2ecef(lat, lon, alt)
    [1115.308, -4844.546, 3982.965]

    """
    # Normal distance from teh surface to the Z axis along the ellipsoid normal
    N = r / np.sqrt(1 - ee**2 * np.sin(lat)**2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - ee**2) * N + alt) * np.sin(lat)

    return np.array([x, y, z])


def site2eci(lat, alt, lst, r=6378.137, ee=8.1819190842622e-2):
    """Calculate the site vector in the IJK coordinate system.

    Author:   C2C Shankar Kulumani   USAFA/CS-19   719-333-4741

    Inputs:
        sitlat - site latitude (radians)
        lst - local sidereal time (radians)
        sitalt - site altitude (meters)

    Outputs:
        R_site - site vector in IJK system

    Globals:
        RE, EEsqrd

    Constants: None

    Coupling: None

    References:
        Astro 321 COMFIX
    """
    N = r / np.sqrt(1 - ee**2 * np.sin(lat)**2)
    x = (N + alt) * np.cos(lat)
    z = ((1 - ee**2) * N + alt) * np.sin(lat)
    eci = np.array([x * np.cos(lst), x * np.sin(lst), z])

    return eci


def ecef2lla(ecef, r=6378.137, ee=8.1819190842622e-2):
    """Converts a ECEF vector to the equivalent Lat, longitude and Altitude
    above the reference ellipsoid

    """
    twopi = 2*np.pi
    tol = 1e-6

    norm_vec = np.linalg.norm(ecef)

    temp = np.sqrt(ecef[0]**2 + ecef[1]**2)
    if temp < tol:
        rtasc = np.sign(ecef[2]) * np.pi * 0.5
    else:
        rtasc = np.arctan2(ecef[1], ecef[0])

    lon = rtasc
    lon = attitude.normalize(lon, 0, 2*np.pi)

    decl = np.arcsin(ecef[2]/ norm_vec)
    latgd = decl

    # iterate to find geodetic latitude
    i = 1
    olddelta = latgd + 10.0
    while np.absolute(olddelta - latgd) >= tol and i < 10:
        oldelta = latgd
        sintemp = np.sin(latgd)
        c = r / np.sqrt(1.0 - ee**2 * sintemp**2)
        latgd = np.arctan2(ecef[2] + c * ee**2 * sintemp, temp)
        i = i +1

    # calculate the height
    if (np.pi / 2 - np.absolute(latgd)) > np.pi/180:
        hellp = temp/np.cos(latgd) - c
    else:
        s = c * (1 - ee**2)
        hellp = ecef[2] / np.sin(latgd) - s

    latgc = gd2gc(latgd, ee**2)
    return latgc, latgd, lon, hellp

def gd2gc(latgd, eesqrd=0.081819221456**2):
    """Only valid for locations on the Earth's surface

    Vallado Example 3-1
    """
    latgc = np.arctan( (1 - eesqrd)*np.tan(latgd))
    return latgc

def gc2gd(latgc, eesqrd=0.081819221456**2):
    """Only valid for locations on the Earth's surface

    Vallado Example 3-1
    """
    latgd = np.arctan(np.tan(latgc) / (1 - eesqrd))
    return latgd

def rhoazel(sat_eci, site_eci, site_lat, site_lst):
    """
    This function calculates the topcentric range,azimuth and elevation from
    the site vector and satellite position vector.

    Author:   C2C Shankar Kulumani   USAFA/CS-19   719-333-4741

    Inputs:
        sat_eci - satellite ECI position vector (km)
        site_eci - site ECI position vector (km)
        site_lat - site geodetic latitude (radians)
        site_lst - site local sidereal time (radians)

    Outputs:
        rho - range (km)
        az - asimuth (radians)
        el - elevation (radians)
    
    Globals: None
    
    Constants: None
    
    Coupling: 

    References:
        Astro 321 Predict LSN 22 
    """

    site2sat_eci = sat_eci - site_eci

    site2sat_sez = attitude.rot3(-site_lst).dot(site2sat_eci)
    site2sat_sez = attitude.rot2(-(np.pi/2 - site_lat)).dot(site2sat_sez)
    
    rho = np.linalg.norm(site2sat_sez)
    el = np.arcsin(site2sat_sez[2] / rho)
    az = attitude.normalize(np.arctan2(site2sat_sez[1], -site2sat_sez[0]), 0, 
            2*np.pi)[0]
# % ------------------------------------------------------------------------------
# %
# %                           function rv2razel
# %
# %  this function converts geocentric equatorial (eci) position and velocity
# %    vectors into range, azimuth, elevation, and rates.  notice the value
# %    of small as it can affect the rate term calculations. the solution uses
# %    the velocity vector to find the singular cases. also, the elevation and
# %    azimuth rate terms are not observable unless the acceleration vector is
# %    available.
# %
# %  author        : david vallado                  719-573-2600   22 jun 2002
# %
# %  revisions
# %    vallado     - add terms for ast calculation                 30 sep 2002
# %    vallado     - update for site fixes                          2 feb 2004 
# %
# %  inputs          description                    range / units
# %    reci        - eci position vector            km
# %    veci        - eci velocity vector            km/s
# %    rs          - eci site position vector       km
# %    latgd       - geodetic latitude              -pi/2 to pi/2 rad
# %    lon         - longitude of site              -2pi to 2pi rad
# %    alt         - altitude                       km
# %    ttt         - julian centuries of tt         centuries
# %    jdut1       - julian date of ut1             days from 4713 bc
# %    lod         - excess length of day           sec
# %    xp          - polar motion coefficient       arc sec
# %    yp          - polar motion coefficient       arc sec
# %    terms       - number of terms for ast calculation 0,2
# %
# %  outputs       :
# %    rho         - satellite range from site      km
# %    az          - azimuth                        0.0 to 2pi rad
# %    el          - elevation                      -pi/2 to pi/2 rad
# %    drho        - range rate                     km/s
# %    daz         - azimuth rate                   rad / s
# %    del         - elevation rate                 rad / s
# %
# %  locals        :
# %    rhoveci     - eci range vector from site     km
# %    drhoveci    - eci velocity vector from site  km / s
# %    rhosez      - sez range vector from site     km
# %    drhosez     - sez velocity vector from site  km
# %    wcrossr     - cross product result           km / s
# %    earthrate   - eci earth's rotation rate vec  rad / s
# %    tempvec     - temporary vector
# %    temp        - temporary real*8 value
# %    temp1       - temporary real*8 value
# %    i           - index
# %
# %  coupling      :
# %    mag         - magnitude of a vector
# %    rot3        - rotation about the 3rd axis
# %    rot2        - rotation about the 2nd axis
# %
# %  references    :
# %    vallado       2007, 268-269, alg 27
# %
# % [rho,az,el,drho,daz,del] = rv2razel ( reci,veci, latgd,lon,alt,ttt,jdut1,lod,xp,yp,terms,ddpsi,ddeps );
# % ------------------------------------------------------------------------------

# function [rho,az,el,drho,daz,del] = rv2razel ( reci,veci, latgd,lon,alt,ttt,jdut1,lod,xp,yp,terms,ddpsi,ddeps );

#     halfpi = pi*0.5;
#     small  = 0.00000001;

#     % --------------------- implementation ------------------------
#     % ----------------- get site vector in ecef -------------------
#     [rs,vs] = site ( latgd,lon,alt );

#     % -------------------- convert eci to ecef --------------------
#     a = [0;0;0];
#     [recef,vecef,aecef] = eci2ecef(reci,veci,a,ttt,jdut1,lod,xp,yp,terms,ddpsi,ddeps);
#     % simplified - just use sidereal time rotation
#     % thetasa= 7.29211514670698e-05 * (1.0  - 0.0/86400.0 );
#     % omegaearth = [0; 0; thetasa;];
#     % [deltapsi,trueeps,meaneps,omega,nut] = nutation(ttt,ddpsi,ddeps);
#     % [st,stdot] = sidereal(jdut1,deltapsi,meaneps,omega,0,0 );
#     %  recef  = st'*reci;
#     %  vecef  = st'*veci - cross( omegaearth,recef );


#     % ------- find ecef range vector from site to satellite -------
#     rhoecef  = recef - rs;
#     drhoecef = vecef;
#     rho      = mag(rhoecef);

#     % ------------- convert to sez for calculations ---------------
#     [tempvec]= rot3( rhoecef, lon          );
#     [rhosez ]= rot2( tempvec, halfpi-latgd );

#     [tempvec]= rot3( drhoecef, lon         );
#     [drhosez]= rot2( tempvec,  halfpi-latgd);

#     % ------------- calculate azimuth and elevation ---------------
#     temp= sqrt( rhosez(1)*rhosez(1) + rhosez(2)*rhosez(2) );
#     if ( ( temp < small ) )           % directly over the north pole
#         el= sign(rhosez(3))*halfpi;   % +- 90 deg
#     else
#         magrhosez = mag(rhosez);
#         el= asin( rhosez(3) / magrhosez );
#     end

#     if ( temp < small )
#         az = atan2( drhosez(2), -drhosez(1) );
#     else
#         az= atan2( rhosez(2)/temp, -rhosez(1)/temp );
#     end

#     % ------ calculate range, azimuth and elevation rates ---------
#     drho= dot(rhosez,drhosez)/rho;
#     if ( abs( temp*temp ) > small )
#         daz= ( drhosez(1)*rhosez(2) - drhosez(2)*rhosez(1) ) / ( temp*temp );
#     else
#         daz= 0.0;
#     end

#     if ( abs( temp ) > small )
#         del= ( drhosez(3) - drho*sin( el ) ) / temp;
#     else
#         del= 0.0;
#     end

    return rho, az, el

def radar_meas(r_sat_eci, v_sat_eci, r_site_eci, site_lat, site_lst):
    """Simulated radar measurements from the site to satellite

    This function will output deterministic radar measurements given a satellite
    position and radar site in the ECI inertial frame.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    Dependencies
    ------------

    Reference
    ---------

    Author
    ------

    """

    # determine transformation from IJK to SEZ

    # determine the range and range rate in SEZ frame

    # compute azimuth and elevation

    # transform velocity to angle rate information
